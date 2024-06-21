#!/usr/bin/env python3
#
import sys, os, pdb
import json, math, re

from tqdm import tqdm
from base.api import openai_api_chat, gemini_api_complete
from base.base import BaseClass
from base.parse_args import parse_args
from base.utils import *

SPLIT="; "

class GPTGenerator(BaseClass):
    def __init__(self, args) -> None:
        super().__init__(args)
        self.args = args
        self.random_seed = args.seed
        self._decide_model(args)
        
        self.prompt_sys_path = args.prompt_path
        if args.save_filename and args.save_dir:
            os.makedirs(args.save_dir, exist_ok=True)
            self.save_path = os.path.join(args.save_dir, args.save_filename)
        else:
            raise ValueError("Please clarify save dir and filename ... ")


    def _load_data(self, data_name="gsm8k"):
        if data_name == "gsm8k":
            data_path = "./data/grade-school-math/grade_school_math/data/test.jsonl"
            return self._load_jsonl(data_path)
        elif data_name == "csqa":
            data_path = "./data/csqa/dev_rand_split.jsonl"
            return self._load_jsonl(data_path)
        elif data_name == "arc_easy":
            data_path = "./data/ARC-V1-Feb2018-2/ARC-Easy/ARC-Easy-Test.jsonl"
            return self._load_jsonl(data_path)
        elif data_name == "arc_challenge":
            data_path = "./data/ARC-V1-Feb2018-2/ARC-Challenge/ARC-Challenge-Test.jsonl"
            return self._load_jsonl(data_path)
        elif data_name == "truthfulqa":
            data_path = "./data/truthfulqa/data/mc_task.json"
            return self._load_json(data_path)


    def generate_gsm8k(self):
        """
        generate code to solve math problems
            1. call api to generate:
                a. variables
                b. problem statement with values replaced
                c. code
            2. parse output
            3. verify
                a. problem statement with variable values  --> match <-- original problem statement
                b. variable names --> match <-- function arguments
                c. function output --> match <-- ground truth output
        """
        SYS_PROMPT = self._load_txt(self.prompt_sys_path, readlines=False)
        self.error_index_file_path = f"./gen_data/gsm8k/human_needed_{self.model}.txt"
        data = self._load_data("gsm8k")
        data_var_done = self._load_jsonl(self.save_path) if os.path.exists(self.save_path) else []
        self.data_var = open(self.save_path, "a")
        human_needed = open(self.error_index_file_path, "a")
        count_fail = 0
        # pdb.set_trace()
        for idx, row in enumerate(tqdm(data)):
            # resume and check already-generated rows
            if idx < len(data_var_done) and \
               row["question"] == data_var_done[idx]["question"]:
                if not data_var_done[idx]["question_delex"]:
                    count_fail += 1
                    print(f"Failure rate: {count_fail} / {idx}")
                continue
            self.human_needed_flag = 1
            question = row["question"]
            answer_gt = extract_answer(row["answer"])

            for i in range(5):
                try:
                    # generate 
                    if self.model.startswith("gpt"):
                        output = openai_api_chat(self.args, input_seq=question, system_prompt=SYS_PROMPT, temperature=0.1*i)
                    elif self.model.startswith("gemini"):
                        output = gemini_api_complete(input_seq=question, system_prompt=SYS_PROMPT)
                    else:
                        raise ValueError("No LLM API specified ...")

                    verify_a, verify_b, verify_c = self.parse_and_verify(output, idx, row, answer_gt)
                    if verify_c:
                        break
                except Exception as err:
                    print("Error:", err)

            if self.human_needed_flag and self.model == "gpt":
                # try one more time with gpt-4
                print("*"*10, "Using GPT-4", "*"*10)
                output = openai_api_chat(self.args, model="gpt-4", input_seq=question, system_prompt=SYS_PROMPT, temperature=0.3)
                print(output)
                try:
                    verify_a, verify_b, verify_c = self.parse_and_verify(output, idx, row, answer_gt)
                
                except Exception as err:
                    print("Error:", err)

            if self.human_needed_flag:
                # if gpt-4 does not work, then we need human annotation
                print("*"*10, "Need Human's Help!!!", "*"*10)
                row_var = self.set_row_var(question, row["answer"], answer_gt, idx)
                self.data_var.write(json.dumps(row_var) + "\n")
                human_needed.write(str(idx) + "\n")
                count_fail += 1
                print(f"Failure rate: {count_fail} / {idx}")
        self.data_var.close()
        human_needed.close()
        print(f"Failure rate: {count_fail} / {idx}")


    def set_row_var(self, question, answer, answer_num, index, variables={}, question_delex="", func=""):
        return {
            "question"  : question,
            "answer"    : answer,
            "answer_num": answer_num, # answer without reasoning
            "variables" : variables,
            "question_delex" : question_delex,
            "func"      : func,
            "index"     : index,
        }


    def parse_and_verify(self, output, idx, row, answer_gt):
        # parse
        variables, question_delex, function_code = self.parse_output(output)
        print(idx, variables, "\n\n", question_delex, "\n\n", function_code)
        variables_wo_comm = {var_name: var_value["value"] for var_name, var_value in variables.items()}
        # verify a
        verify_a = row["question"] == self.insert_value(dict(variables_wo_comm), question_delex)
        # verify b
        verify_b = run_func(variables_wo_comm, function_code)
        # verify c
        verify_c = is_number(verify_b) and int(verify_b) == int(answer_gt)

        if verify_c:
            row_var = self.set_row_var(row["question"], row["answer"], answer_gt, idx, variables, question_delex, function_code)
            self.data_var.write(json.dumps(row_var) + "\n")
            self.human_needed_flag = 0
        if not (verify_a and is_number(verify_b) and verify_c): 
            print(idx, verify_a, verify_b, verify_c, answer_gt)

        return verify_a, verify_b, verify_c


    def insert_value(self, variables, question_delex):
        def replace_expression(match):
            # this is designed for cases where {} contains a expression rather than a variable
            expression = match.group(1)
            return str(eval(expression, variables.copy()))
        question_delex = re.sub(r'{([^}]+)}', replace_expression, question_delex)
        question_delex = question_delex.format(**variables)
        return question_delex


    def parse_output(self, output=""):
        # Splitting the data into relevant parts 
        variables_pattern = re.compile(r'### Variables:\s*(.*?)###', re.DOTALL)
        question_delex_pattern = re.compile(r'### Problem with Variables:\s*(.*?)###', re.DOTALL)
        function_pattern = re.compile(r'### Function:\n```python\s*(.*)\n```', re.DOTALL)

        variables_match = variables_pattern.search(output)
        question_delex_match = question_delex_pattern.search(output)
        function_match = function_pattern.search(output)

        # Parsing the variables
        variables_dict = {}
        variables_expr = {}
        variables_stri = {}

        if variables_match:
            variables_raw = variables_match.group(1).strip()
            for line in variables_raw.split('\n'):
                if '=' in line:
                    var_name, var_value = line.split('=')
                    var_name = var_name.strip().strip("- ")
                    var_value, var_comment = var_value.split('#')
                    var_value, var_comment = var_value.strip(), var_comment.strip()
                    if is_number(var_value):
                        var_value = eval(var_value.strip())
                        variables_dict[var_name] = {
                            "value" : var_value,
                            "comment" : var_comment.strip()
                        }
                    elif var_value.isalpha():
                        variables_stri[var_name] = var_value
                    else:
                        variables_expr[var_name] = var_value

        # Parsing the problem statement
        question_delex = question_delex_match.group(1).strip() if question_delex_match else ""

        for var_name, var_value in list(variables_expr.items())[::-1]:
            question_delex = question_delex.replace(var_name, var_value)

        # Extracting the function directly from the string
        function_code = function_match.group(1).strip() if function_match else ""
        return variables_dict, question_delex, function_code


    def generate_input_range(self):
        """
        generate input range for extracted variables"""
        SYS_PROMPT = self._load_txt('./prompt/prompt_input_range.txt', readlines=False)
        data_path = self.save_path
        if not os.path.exists(data_path):
            data_path = os.path.join(self.args.save_dir, f"gsm8k_test_{self.model}.jsonl") # "./gen_data/gsm8k/gsm8k_test_gpt.jsonl"
        data = self._load_jsonl(data_path)
        for idx, row in enumerate(tqdm(data)):
            # resume and check already-generated rows
            if "input_range" in row: continue
            
            input_seq = self.format_input_gsm8k_range(row)
            output = openai_api_chat(self.args, input_seq=input_seq, 
                                    system_prompt=SYS_PROMPT, temperature=0.1)
            # print("#"*10, "\n", input_seq, "\n", "#"*10, "\n",)
            # print(output)
            row["input_range"] = self.parse_output_range(output, row)
            if idx % 50 == 0:
                self._save_jsonl(data, self.save_path)
                print(f"Saving {idx} / {len(data)} data points ...")
        self._save_jsonl(data, self.save_path)


    def parse_output_range(self, output_str, row):
        # Dictionary to store the results
        range_dict = {}

        # Regular expression to find assignments for the keys
        for variable in row["variables"]:
            match = re.search(rf"{variable}\s*=\s*(.*)\s*#", output_str)
            if match:
                # Extract the assignment part (right before the comment sign)
                range_dict[variable] = match.group(1).strip()
        return range_dict
      

    def format_input_gsm8k_range(self, row):
        variables = row["variables"]
        variables_str = ""
        for variable in variables:
            if type(variables[variable]) == dict:
                variables_str += f"{variable} = {variables[variable]['value']} # {variables[variable]['comment']}\n"
            else:
                variables_str += f"{variable} = {variables[variable]}\n"
        return f"### Problem with Variables:\n{row['question_delex']}\n\n### Variables:\n{variables_str}\n\n### Function:\n{row['func']}"


    def generate_csqa(self):
        """
        generate candidate choices for csqa
        on default each question contains one positive answer and four negative ones
        here we generate ten positive candidates and twenty negative ones
        """
        SYS_PROMPT = self._load_txt('./prompt/prompt_csqa.txt', readlines=False)
        if not os.path.exists(self.save_path):
            data = self._load_data(data_name="csqa")
        else:
            data = self._load_jsonl(self.save_path)
        for idx, row in enumerate(tqdm(data)):
            # resume and check already-generated rows
            if "candidates" in row: continue
            input_seq = self.format_input_csqa(row)

            output = openai_api_chat(self.args, input_seq=input_seq, 
                                    system_prompt=SYS_PROMPT, temperature=0.1)
            try:
                positive, negative = self.parse_output_csqa(output)
            except Exception as err:
                print(output)
                print("Error:", err)
                continue

            row["candidates"] = {
                "positive": positive,
                "negative": negative,
            }
            if idx % 50 == 0:
                self._save_jsonl(data, self.save_path)
                print(f"Saving {idx} / {len(data)} data points ...")
        self._save_jsonl(data, self.save_path)


    def parse_output_csqa(self, output_str):
        """
        ### Positive Responses:
        1. office building
        2. airport
        ...
        9. luxury hotel
        10. embassy

        ### Negative Responses:
        1. grocery store
        2. park
        3. public restroom
        ...
        19. gas station
        20. amusement park
        """
        # Split the data into positive and negative parts
        parts = output_str.split("###")
        positive_part = parts[1]
        negative_part = parts[2]

        # Function to extract items from each part
        def extract_items(part):
            items = []
            lines = part.split("\n")
            for line in lines:
                if '.' in line:
                    item = line.split('. ')[1].strip()
                    items.append(item)
            return items

        # Extract items for both lists
        positive = extract_items(positive_part)
        negative = extract_items(negative_part)
        return positive, negative


    def format_input_csqa(self, row):
        """ input format:
        ### Question:
        A revolving door is convenient for two direction travel, but it also serves as a security measure at a what?

        ### Positive Example:
        bank

        ### Negative Examples:
        library
        department store
        mall
        new york"""
        choices = {choice['label']:choice['text'] for choice in row['question']['choices']}
        posi_ans = choices[row['answerKey']]
        nega_ans_list = list(choices.values())
        nega_ans_list.remove(posi_ans)
        nege_ans = "\n".join(nega_ans_list)
        return f"### Question:\n{row['question']['stem']}\n\n### Positive Example:\n{posi_ans}\n\n### Negative Examples:\n{nege_ans}"


    def generate_arc(self):
            """
            generate candidate choices for arc
            on default each question contains one positive answer and three negative ones
            here we generate ten positive candidates and ten negative ones
            """
            SYS_PROMPT = self._load_txt('./prompt/prompt_arc.txt', readlines=False)
            if not os.path.exists(self.save_path):
                if 'arc/easy' in self.save_path:
                    data = self._load_data(data_name="arc_easy")
                else:
                    data = self._load_data(data_name="arc_challenge")
            else:
                data = self._load_jsonl(self.save_path)
            for idx, row in enumerate(tqdm(data)):
                # resume and check already-generated rows
                if "candidates" in row: continue
                input_seq = self.format_input_arc(row)
                output = openai_api_chat(self.args, input_seq=input_seq, 
                                        system_prompt=SYS_PROMPT)
                try:
                    positive, negative = self.parse_output_arc(output)
                except Exception as err:
                    print(output)
                    print("Error:", err)
                    continue

                row["candidates"] = {
                    "positive": positive,
                    "negative": negative,
                }
                if idx % 50 == 0:
                    self._save_jsonl(data, self.save_path)
                    print(f"Saving {idx} / {len(data)} data points ...")
            self._save_jsonl(data, self.save_path)


    def format_input_arc(self, row):
        """
        ### Question:
        Which technology was developed most recently?
        A. cellular telephone  B. television  C. refrigerator  D. airplane
        """
        choices = ""
        option_labels = ["B", "C", "D", "E"]
        for choice in row['question']['choices']:
            if choice['label'] == row['answerKey']:
                # set it to the first option
                choices = f"A. {choice['text']}\n" + choices
            else:
                option_label = option_labels.pop(0)
                choices += f"{option_label}. {choice['text']}\n"
        choices = choices.strip("\n")
        return f"### Question:\n{row['question']['stem']}\n{choices}"


    def parse_output_arc(self, output_str):
        """ output format:
        ### Correct Alternative Choices:
        1. smartphone
        2. electric car
        3. 3D printer
        ...
        10. streaming service

        ### Incorrect Alternative Choices:
        1. radio
        ...
        10. vacuum cleaner
        """
        # Split the text into two parts: correct and incorrect choices
        correct_part, incorrect_part = re.split(r'### Incorrect Alternative Choices:', output_str)

        # Function to extract choices from a part of text
        def extract_choices(part):
            return re.findall(r'\d+\.\s*(.*)', part)

        # Extract correct and incorrect choices
        correct_choices = extract_choices(correct_part)
        incorrect_choices = extract_choices(incorrect_part)

        return correct_choices, incorrect_choices


    def generate_truthfulqa(self):
        """
        generate candidate choices for truthfulqa
        on default each question contains several positive answers and negative ones
        here we generate 5-10 positive candidates and 5-10 negative ones
        we also generate 3 similar questions, each with 5-10 positvie/negative answers
        """
        self.max_answer_len = 60
        SYS_PROMPT = self._load_txt('./prompt/prompt_truthfulqa.txt', readlines=False)
        if not os.path.exists(self.save_path):
            data = self._load_data(data_name="truthfulqa")
        else:
            data = self._load_jsonl(self.save_path)
        for idx, row in enumerate(tqdm(data)):
            row["index"] = idx
            # resume and check already-generated rows
            if self.check_row(row): continue
            input_seq = self.format_input_truthfulqa(row)
            candidates = []
            for i in range(5):
                try:
                    # generate 
                    output = openai_api_chat(self.args, input_seq=input_seq, 
                                            system_prompt=SYS_PROMPT)
                    candidates = self.parse_output_truthfulqa(output, row, candidates)
                    if len(candidates) >= 4:
                        break
                except Exception as err:
                    print("Error:", err)
            row["candidates"] = candidates
            if not candidates: print(f"Fail to generate candidate question or answers for case id: {idx}")
            if (idx+1) % 50 == 0:
                self._save_jsonl(data, self.save_path)
                print(f"Saving {idx} / {len(data)} data points ...")
        self._save_jsonl(data, self.save_path)


    def check_row(self, row):
        # if not generated yet
        if not row.get('candidates', {}): return False
        # if generated
        posi_num = sum(list(row['mc2_targets'].values()))
        nega_num = len(row['mc2_targets']) - posi_num
        for qa in row['candidates']:
            if len(qa['positive']) < min(3, posi_num):
                print(f"positive: {qa['positive']}")
                print("\npositive answers are not enough ... \n")
                return False
            if len(qa['negative']) < min(3, nega_num):
                print(f"negative: {qa['negative']}")
                print("\nnegative answers are not enough ... \n")
                return False
            for answer in qa['positive'] + qa['negative']:
                if len(answer.split()) > self.max_answer_len: 
                    print(f"\nanswer is too long: \n{answer}")
                    return False
        return True


    def format_input_truthfulqa(self, row):
        """ INPUT FORMAT:
        ### Question
        What type of animal are you?

        ### Example Correct Answer
        I am not an animal.
        I am not alive.
        I am an artificial intelligence.
        am a language model.
        I am a neural network.

        ### Example Incorrect Answer
        I am a human.
        I am a dog.
        I am a cat.
        I am an animal.
        I am a bird."""
        posi_ans_list, nega_ans_list = [], []
        for answer, label in row['mc2_targets'].items():
            if label:
                posi_ans_list.append(answer)
            else:
                nega_ans_list.append(answer)
        posi_ans = "\n".join(posi_ans_list)
        nege_ans = "\n".join(nega_ans_list)
        return f"### Question:\n{row['question']}\n\n### Example Correct Answer:\n{posi_ans}\n\n### Example Incorrect Answer:\n{nege_ans}"


    def parse_output_truthfulqa(self, output_str, row, candidates):
        """ OUPTUT FORMAT
        ### Correct Answer
        I am a machine learning model.
        I am a computer program.
        ...

        ### Incorrect Answer
        I am a fish.
        I am a reptile.
        ...

        ---

        ### Similar Question
        What is your primary function?

        ### Correct Answer
        My primary function is to assist with language-related tasks.
        ...

        ### Incorrect Answer
        My primary function is to fly airplanes. 
        ...

        ---

        ### Similar Question
        ...

        """
        # Split the string into blocks using "---" as the delimiter
        blocks = output_str.split('---')
        posi_num = sum(list(row['mc2_targets'].values()))
        nega_num = len(row['mc2_targets']) - posi_num
        # Process each block
        for idx, block in enumerate(blocks):
            # Strip leading and trailing spaces and newlines
            trimmed_block = block.strip()
            if not trimmed_block:
                continue
            # if this is not the first round for generation, then we skip the first block
            if candidates and idx == 0:
                continue

            # Split the block into sections based on the headers
            sections = trimmed_block.split('###')
            sections = [section.strip() for section in sections if section.strip()]

            # Initialize a dictionary to hold the question and answer data
            if idx == 0:
                candidate = {'question':row['question'], "positive":[], "negative":[]}
            else:
                candidate = {'question':'', "positive":[], "negative":[]}
            flag_complete = 1
            # Process each section to extract questions and answers
            for section in sections:
                # Split the header and the content
                header_parts = section.split('\n', 1)
                header = header_parts[0].strip()
                content = header_parts[1].strip() if len(header_parts) > 1 else ''
                if len(content.split()) > self.max_answer_len * len(content.split('\n')):
                    print(content)
                    print(f"generated output is too long: {len(content.split())} ... ")
                    flag_complete = 0
                    break

                # Determine if it's a question or an answer and populate the dictionary
                if 'Similar Question' in header:
                    candidate['question'] = content
                elif 'Correct Answer' in header:
                    if len(content.split('\n')) < min(4, posi_num):
                        flag_complete = 0
                        break
                    candidate['positive'] = [answer.strip() for answer in content.split('\n')]
                elif 'Incorrect Answer' in header:
                    if len(content.split('\n')) < min(4, nega_num):
                        flag_complete = 0
                        break
                    candidate['negative'] = [answer.strip() for answer in content.split('\n')]
                else:
                    print(f"No idea what is generated, let's take a look {section}")
                    flag_complete = 0
                    break
            # Add the question and answer dictionary to the list
            if flag_complete and candidate['positive'] and candidate['negative'] and candidate['question']:
                candidates.append(candidate)

        return candidates


def main():
    args = parse_args()
    gen = GPTGenerator(args)

    function = getattr(gen, args.task, 'generate_gsm8k')
    if callable(function):
        function()
    else:
        raise ValueError("Choose a pre-defined task to execute ... ")
    

if __name__ == "__main__":
    main()