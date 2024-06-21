#!/usr/bin/env python3
#
import sys, os, pdb
import json, math, re, csv
import random
import signal
import pandas as pd
import numpy as np

from tqdm import tqdm
from collections import defaultdict
from base.base import BaseClass
from base.parse_args import parse_args
from base.utils import *

SPLIT="; "

class GPTAnnotator(BaseClass):
    """
    This class is designed for 
        1. sampling new values
        2. computing corresponding ground-truth result 
        3. generating test set
        4. copying a training set and match column keys with test set"""
    def __init__(self, args) -> None:
        super().__init__(args)
        self.args = args
        self.random_seed = args.seed
        self._decide_model(args)
       

    def generate_annotation_set_for_code_gsm8k(self):
        """
        load problem statment template, sample three set new values and output"""
         
        self.data_path = f"./gen_data/gsm8k/gsm8k_test_{self.model}.jsonl"
        data = self._load_jsonl(self.data_path)
        output_file_path="./gen_data/gsm8k_annotation_gpt.csv"
        # Open the input JSONL file and the output CSV file
        with open(output_file_path, 'w', newline='') as csv_file:
            csv_writer = csv.writer(csv_file)
            # Write the header row to the CSV file
            csv_writer.writerow(['index', 'question_original', 'question_delex', 'code',
                            'input_1', 'input_1_modified', 'question_1', 'output_1', 'output_1_code', 
                            'input_2', 'input_2_modified', 'question_2', 'output_2', 'output_2_code', 
                            'input_3', 'input_3_modified', 'question_3', 'output_3', 'output_3_code',
                            "code_modified", 'output_1_modified', 'output_1_modified', 'output_1_modified'])
            
            # Read each line from the JSONL file
            for row in data:
                # Extract required fields
                index = row.get('index', '')
                question = row.get('question', '')
                question_delex = row.get('question_delex', '')
                func = row.get('func', '')
                write_row = [index, question, question_delex, func]
                for _ in range(3):
                    variables = {var:self.sample_value(value) for var, value in row["variables"].items()}
                    question_refill = self.insert_value(variables, row["question_delex"])
                    answer = run_func(variables, row["func"])
                    write_row += [variables, '', question_refill, '', answer,]
                write_row += ['', '', '', '',]
                
                # Write the data row to the CSV file
                csv_writer.writerow(write_row)


    def generate_annotation_set_for_range_gsm8k(self):
        """
        load problem statment template along with range
        generate csv for verify input range, and fix some errors from generate_annotation_set_for_code()"""
        data = self._load_jsonl("./gen_data/gsm8k/gsm8k_test_gpt_human_range.jsonl")
        output_file_path="./gen_data/gsm8k/gsm8k_annotation_range_gpt.csv"
        # Open the input JSONL file and the output CSV file
        with open(output_file_path, 'w', newline='') as csv_file:
            csv_writer = csv.writer(csv_file)
            # Write the header row to the CSV file
            csv_writer.writerow(['index', 'question', 'question_delex', 'question_delex_modified', 'code',
                            "code_modified","answer_complete", "answer", "computed_answer", "variables", "input_range", "input_range_modified"])
            count = 0
            # Read each line from the JSONL file
            for row in tqdm(data):
                # Extract required fields
                index = row.get("index", "")
                question = row.get("question", "")
                question_delex = row.get("question_delex", "")
                func = row.get("func", "")
                answer_complete = row.get("answer", "")
                answer = row.get("answer_num", "")
                variables = row.get("variables", "")
                input_range = row.get("input_range", "")
                write_row = [index, question, question_delex, '', func, '', answer_complete]
                
                # # # verify code
                if not variables:
                    computed_answer = ''
                elif type(list(variables.values())[0]) == dict:
                    variables_wo_comm = {var_name: var_value["value"] for var_name, var_value in variables.items()}
                    computed_answer = run_func(variables=variables_wo_comm, function_code=func)
                else:
                    variables_wo_comm = variables
                    computed_answer = run_func(variables=variables_wo_comm, function_code=func)
                if is_number(computed_answer) and int(computed_answer) == int(answer):
                    write_row += ['', '']
                else:
                    write_row += [answer, computed_answer]
                    # if computed_answer:
                    print(row["index"])
                    count += 1
                write_row += [str(variables), str(input_range), '']
                # Write the data row to the CSV file
                csv_writer.writerow(write_row)
            print(count)
                

    def integrate_annotation(self):
        """
        This function clean up human check for generated func and question_delex"""
        data_human = self._load_csv("./gen_data/gsm8k/gsm8k_annotation_human.csv")
        data_ori = self._load_jsonl("./gen_data/gsm8k/gsm8k_test_gpt.jsonl")
        count_delex, count_code = 0, 0
        for i in tqdm(range(len(data_ori))):
            assert data_ori[i]["index"] == i, "index mismatch for original data"
            assert data_human["index"][i] == i, "index mismatch for human annotation data"
            assert data_ori[i]["question"] == data_human["question_original"][i].strip('"'), "original question mismatch"
            # delexicalized question statement is modified
            if type(data_human["question_delex_modified"][i]) == str and data_human["question_delex_modified"][i]:
                data_ori[i]["question_delex"] = data_human["question_delex_modified"][i].strip('"')
            # func code is modified
            if type(data_human["code_modified"][i]) == str and data_human["code_modified"][i]:
                data_ori[i]["func"] = data_human["code_modified"][i].strip('"')
            # the empty delexicalized question statement is filled 
            if not data_ori[i]["question_delex"]:
                if type(data_human["question_delex"][i]) == str and data_human["question_delex"][i]:
                    data_ori[i]["question_delex"] = data_human["question_delex"][i].strip('"')
                else:
                    count_delex += 1
            # the empty function code is filled 
            if not data_ori[i]["func"]:
                if type(data_human["code"][i]) == str and data_human["code"][i]:
                    data_ori[i]["func"] = data_human["code"][i].strip('"')
                else:
                    count_code += 1


        print(count_delex, count_code,)
        self._save_jsonl(data_ori, "./gen_data/gsm8k/gsm8k_test_gpt_human.jsonl")


    def integrate_input_range(self):
        """
        This functino integrate input range to original function
        The input range should satisfy:
            1. not a constant number
            2. include the original value
            3. match variables in question_delext"""
        data_ori = self._load_jsonl("./gen_data/gsm8k/gsm8k_test_gpt_human.jsonl")
        data_range = self._load_jsonl("./gen_data/gsm8k/gsm8k_test_gpt_range.jsonl")
        for i in tqdm(range(len(data_ori))):
            assert data_ori[i]["index"] == i, f"index {i} mismatch for original data"
            assert data_range[i]["index"] == i, "index mismatch for input range data"
            assert data_ori[i]["question"] == data_range[i]["question"], "original question mismatch"
            data_ori[i]["input_range"] = {}
            for variable in data_range[i]["input_range"]:
                # skip constant number
                if is_number(data_range[i]["input_range"][variable]):
                    continue
                # skip if variable not in delexicalized question
                if variable not in data_ori[i]["question_delex"]:
                    continue
                data_ori[i]["input_range"][variable] = data_range[i]["input_range"][variable]
        self._save_jsonl(data_ori, "./gen_data/gsm8k/gsm8k_test_gpt_human_range.jsonl")


    def integrate_annotation_range(self):
        """
        This function clean up human check for input range"""
        data_human = self._load_csv("./gen_data/gsm8k/gsm8k_annotation_range_human.csv")
        data_ori = self._load_jsonl("./gen_data/gsm8k/gsm8k_test_gpt_human_range.jsonl")
        for i in (range(len(data_human))):
            # pdb.set_trace()
            assert data_ori[i]["index"] == i, f"index {i} mismatch for original data {data_ori[i]['index']}"
            assert int(data_human["index"][i]) == i, f"index {i} mismatch for human annotation data {data_human['index'][i]}"
            assert data_ori[i]["question"] == data_human["question"][i].strip('"'), "original question mismatch"
            # delexicalized question statement is modified
            if type(data_human["question_delex_modified"][i]) == str and data_human["question_delex_modified"][i]:
                data_ori[i]["question_delex"] = data_human["question_delex_modified"][i].strip('"')
            else:
                data_ori[i]["question_delex"] = data_human["question_delex"][i].strip('"')
            # verify if variables in delexicalized question match function input arguments
            var_question = self.extract_question_placeholder(data_ori[i]["question_delex"])


            # variables are modified
            if type(data_human["variables_modified"][i]) == str and data_human["variables_modified"][i]:
                variables = eval(data_human["variables_modified"][i])
            else:
                variables = eval(data_human["variables"][i])
            # filter variables according input arguments
            variables_filter = {}
            for key_, value_ in variables.items():
                key_ = key_.strip()
                # if key_ not in var_question: continue
                if type(value_) == dict:
                    variables_filter[key_] = value_
                else:
                    variables_filter[key_] = {
                        "value": value_,
                        "comment": ""
                    }
            data_ori[i]["variables"] = variables_filter
            # missing_var = set(var_question) - set(variables_filter.keys())
            # if missing_var:
            #     print(f"ID: {data_human['index'][i]}, " + ", ".join(missing_var) + "\n")

            # func code is modified
            if type(data_human["code_modified"][i]) == str and data_human["code_modified"][i]:
                data_ori[i]["func"] = data_human["code_modified"][i].strip('"')
            else:
                data_ori[i]["func"] = data_human["code"][i].strip('"')
            # # verify func get the same result as ground truth
            # variables_wo_comm = {var_name: var_value["value"] for var_name, var_value in variables_filter.items()}
            # result = run_func(variables_wo_comm, data_ori[i]["func"])
            # if is_number(result) and int(result) != int(data_ori[data_human['index'][i]]["answer_num"]):
            #     print(f"ID: {data_human['index'][i]}, {result}, {data_ori[data_human['index'][i]]['answer_num']} \n")

            # input range are modified
            if type(data_human["input_range_modified"][i]) == str and data_human["input_range_modified"][i]:
                input_range = eval(data_human["input_range_modified"][i])
            else:
                input_range = eval(data_human["input_range"][i])
            # filter input_range according input arguments
            data_ori[i]["input_range"] = self.complete_range_wi_original_value(input_range, variables_filter)
            self.verify_input_range(data_human["index"][i], var_question, data_ori[i]["input_range"], data_ori[i]["func"])

        self._save_jsonl(data_ori, "./gen_data/gsm8k/gsm8k_test_gpt_human_range_human.jsonl")


    def complete_range_wi_original_value(self, input_range, variables_ori):
        # input_range = {key_:value_ for key_, value_ in input_range.items() if key_ in variables_ori}
        for key_, value_ in variables_ori.items():
            if key_ not in input_range:
                input_range[key_] = str(value_["value"])
        return input_range


    def verify_input_range(self, idx, var_question, input_range, func):
        """
        this function is designed to verify if input_range is executable
        and compatible with question_delex and func code"""
        # compatible with question_delex            
        missing_var = set(var_question) - set(input_range.keys())
        if missing_var:
            print(f"ID: {idx}, " + ", ".join(missing_var) + "\n")
        
        for seed in range(50):
            self._set_seed(seed)
            # check whether executable
            try:
                sample_values = self.generate_sample_values(input_range)
                # print(sample_values)

                # # compatible with func
                result = run_func(sample_values, func)
                if not is_number(result):
                    print(f"ID: {idx}, {result} \n")
                    break
                if result < 0:
                    print(f"ID: {idx}, lead to answer of negative number: {result}")
                    break
            except Exception as err:
                print(f"ID: {idx}", str(err))
                break


    def generate_sample_values(self, variable_dict):
        # Create a copy of the variable dictionary to store the generated values
        generated_values = variable_dict.copy()

        # Create a dependency graph using a defaultdict
        graph = defaultdict(list)
        vertices = []
        for variable, value_range in variable_dict.items():
            vertices.append(variable)
            for v in variable_dict:
                if v in value_range and v != variable:
                    graph[v].append(variable)

        # Perform topological sorting to get the evaluation order
        evaluation_order = topological_sort(graph, vertices)

        # Evaluate the variables in the obtained order
        for variable in evaluation_order:
            value_range = variable_dict[variable]
            locals()[variable] = eval(value_range)
            generated_values[variable] = locals()[variable]
        return generated_values


    def extract_question_placeholder(self, question_delex):
        known_functions = ['abs', 'min', 'max', 'sum', 'len']  # Default list of known functions to ignore
    
        variables = set()
        # Regular expression to capture potential variable names within the placeholders
        pattern = re.compile(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b')  # Matches variable-like patterns
        
        placeholders = re.findall(r'\{(.*?)\}', question_delex)
        for placeholder in placeholders:
            # Extract all possible variable-like substrings
            found_variables = pattern.findall(placeholder)
            for var in found_variables:
                if var not in known_functions and not var.isdigit():  # Ensure it's not a known function or a digit
                    variables.add(var)
        return variables


    def extract_function_arguments(self, code):
        # Split the function definition line to isolate the part within the parentheses
        start = code.find("def solution(") + len("def solution(")
        end = code.find("):", start)
        arguments = code[start:end]
        
        # Split the arguments by commas and strip any surrounding spaces
        argument_list = [arg.strip() for arg in arguments.split(',')]
        return argument_list


    def generate_annotation_set_for_gsm8k_llama3i(self):
        """
        convert results of llama3i on gsm8k to csv
        to verify if it is the model that cannot handle the questions
        consider cases where llama3i can handle in gsm but not gsm_var"""
        results_ori_path = "result/gsm8k/5_shot/llama3i_gsm8k_bs32/meta-llama__Meta-Llama-3-8B-Instruct/samples_gsm8k_2024-06-08T05-42-41.351110.json"
        results_var_path = "result/gsm8k/5_shot/llama3i_gsm8k_var_40_bs32/meta-llama__Meta-Llama-3-8B-Instruct/samples_gsm8k_var_40_2024-06-09T22-28-21.331942.json"
        output_file_path="./gen_data/gsm8k/results_annotation_llama3i_var_40_bs32.csv"
        results_ori = self._load_jsonl(results_ori_path)
        results_var = self._load_jsonl(results_var_path)
        count_round, count_err = 0, 0
        # Open the input JSONL file and the output CSV file
        with open(output_file_path, 'w', newline='') as csv_file:
            csv_writer = csv.writer(csv_file)
            # Write the header row to the CSV file
            csv_writer.writerow(['index', 'question', 'ground_truth_reasoning', 
                    'ground_truth_answer', 'extracted_answer_from_model', 'complete_response_from_model'])
            # Read each line from the JSONL file
            for idx, row in enumerate(tqdm(results_var)):
                # ignore successful case
                if row["exact_match"]: continue
                # ignore when results_ori also fails
                if not results_ori[idx]["exact_match"]: continue
                index = row["doc"]["index"]
                question = row["doc"]["question"]
                answer_gt = row["doc"]["answer"].strip("#### ")
                answer_gt_full = results_ori[idx]["target"]
                answer_model = row["filtered_resps"]
                answer_model_full = row["resps"][0][0]

                if answer_model[0] != "[invalid]":
                    answer_model_num = int(float(answer_model[0].replace(',','').strip('$.')))
                    if int(float(answer_gt)) - 1 <= answer_model_num <= int(float(answer_gt)) + 1:
                        count_round += 1
                        continue
                if str(answer_gt) in answer_model_full:
                    count_round += 1
                    continue

                write_row = [index, question, answer_gt_full, answer_gt, answer_model, 
                            answer_model_full]
                count_err +=1

                # Write the data row to the CSV file
                csv_writer.writerow(write_row)
        print(f"{count_round} out of {len(results_ori)} cases are almost correct ...")
        print(f"{count_err} out of {len(results_ori)} cases are wrong ...")


    def generate_annotation_set_for_csqa(self):
        """
        load problem statment template along with range
        generate csv for verify input range, and fix some errors from generate_annotation_set_for_code()"""
        data = self._load_jsonl("./gen_data/csqa/dev_gpt4o.jsonl")
        output_file_path="./gen_data/csqa/dev_annotation_gpt4o.csv"
        # Open the input JSONL file and the output CSV file
        with open(output_file_path, 'w', newline='') as csv_file:
            csv_writer = csv.writer(csv_file)
            # Write the header row to the CSV file
            csv_writer.writerow(['index', 'question', 'positive_choices', 'negative_choices', 
                    'positive_candidates', 'positive_candidates_modified', 'negative_candidates', 'negative_candidates_modified'])
            # Read each line from the JSONL file
            for idx, row in enumerate(tqdm(data)):
                # Extract required fields
                index = idx
                question = row["question"]["stem"]

                choices = {choice['label']:choice['text'] for choice in row['question']['choices']}
                posi_choices = choices[row['answerKey']]
                nega_choices = list(choices.values())
                nega_choices.remove(posi_choices)

                posi_candidates = row['candidates']['positive']
                nega_candidates = row['candidates']['negative']

                nega_choices = "\n".join(nega_choices)
                posi_candidates = "\n".join(posi_candidates)
                nega_candidates = "\n".join(nega_candidates)
                
                write_row = [index, question, posi_choices, nega_choices, 
                            posi_candidates, '', nega_candidates, '']

                # Write the data row to the CSV file
                csv_writer.writerow(write_row)


    def integrate_annotation_set_for_csqa(self):
        """
        This function clean up human check for csqa
        based on positive_candidates_modified and negative_candidates_modified
        if positive_candidates_modified have content, then replace positive_candidates with it
        if positive_candidates_modified is empty, then use positive_candidates (generated choices)
        if positive_candidates_modified is none, do not positive_choices (original choices)
        """
        data_human = self._load_csv("gen_data/csqa/dev_annotation_gpt4o_human.csv")
        data_ori = self._load_jsonl("gen_data/csqa/dev_gpt4o.jsonl")
        count = {"positive": {"modify":0, "remove":0}, "negative": {"modify":0, "remove":0}}
        for i in tqdm(range(len(data_ori))):
            # assert data_ori[i]["index"] == i, "index mismatch for original data" # original data does not include index
            assert data_human["index"][i] == i, "index mismatch for human annotation data"
            assert data_ori[i]["question"]["stem"] == data_human["question"][i].strip('"'), f"original question mismatch for id {i}"
            data_ori[i]["index"] = i
            for mode in ["positive", "negative"]:
                if data_human[f"{mode}_candidates_modified"][i] == "none":
                    data_ori[i]["candidates"][mode] = []
                    count[mode]["remove"] += 1
                elif type(data_human[f"{mode}_candidates_modified"][i]) == str:
                    if type(data_human[f"{mode}_candidates_modified"][i]) != str: pdb.set_trace()
                    data_ori[i]["candidates"][mode] = data_human[f"{mode}_candidates_modified"][i].split("\n")
                    count[mode]["modify"] += 1
                elif not np.isnan(data_human[f"{mode}_candidates_modified"][i]):
                    raise ValueError(f"Error case with id {i}: ", data_human[f"{mode}_candidates_modified"][i])
        print(f"{count['positive']['remove']}+{count['positive']['modify']} positive and {count['negative']['remove']}+{count['negative']['modify']} negative out of {len(data_ori)} generated candidates are modified ... ")
        self._save_jsonl(data_ori, "./gen_data/csqa/dev_gpt4o_human.jsonl")


    def generate_annotation_set_for_arc(self):
        """
        load problem statment template along with range
        generate csv for verify input range, and fix some errors from generate_annotation_set_for_code()"""
        data = self._load_jsonl("./gen_data/arc/challenge/test_gpt4o.jsonl")
        output_file_path="./gen_data/arc/challenge/test_annotation_gpt4o.csv"
        # Open the input JSONL file and the output CSV file
        with open(output_file_path, 'w', newline='') as csv_file:
            csv_writer = csv.writer(csv_file)
            # Write the header row to the CSV file
            csv_writer.writerow(['index', 'question', 'positive_choices', 'negative_choices', 
                    'positive_candidates', 'positive_candidates_modified', 'negative_candidates', 'negative_candidates_modified'])
            # Read each line from the JSONL file
            for idx, row in enumerate(tqdm(data)):
                # Extract required fields
                index = idx
                question = row["question"]["stem"]

                choices = {choice['label']: choice['text'] for choice in row['question']['choices']}
                posi_choices = choices[row['answerKey']]
                nega_choices = list(choices.values())
                nega_choices.remove(posi_choices)

                posi_candidates = row['candidates']['positive']
                nega_candidates = row['candidates']['negative']

                nega_choices = "\n".join(nega_choices)
                posi_candidates = "\n".join(posi_candidates)
                nega_candidates = "\n".join(nega_candidates)
                
                write_row = [index, question, posi_choices, nega_choices, 
                            posi_candidates, '', nega_candidates, '']

                # Write the data row to the CSV file
                csv_writer.writerow(write_row)


    def generate_annotation_set_for_arc_llama3i(self):
        """
        convert results of llama3i on gsm8k to csv
        to verify if it is the model that cannot handle the questions
        consider cases where llama3i can handle in gsm but not gsm_var"""
        results_ori_path = "result/gsm8k/5_shot/llama3i_gsm8k_bs32/meta-llama__Meta-Llama-3-8B-Instruct/samples_gsm8k_2024-06-08T05-42-41.351110.json"
        results_var_path = "result/gsm8k/5_shot/llama3i_gsm8k_var_40_bs32/meta-llama__Meta-Llama-3-8B-Instruct/samples_gsm8k_var_40_2024-06-09T22-28-21.331942.json"
        output_file_path="./gen_data/gsm8k/results_annotation_llama3i_var_40_bs32.csv"

        # results_ori_path = "result/arc_challenge/25_shot/llama3i_arc_challenge_bs16/meta-llama__Meta-Llama-3-8B-Instruct/samples_arc_challenge_2024-06-10T22-34-27.647570.json"
        # results_var_path = "result/arc_challenge/25_shot/llama3i_arc_challenge_40_bs16/meta-llama__Meta-Llama-3-8B-Instruct/samples_arc_challenge_40_2024-06-11T19-46-10.509386.json"
        # output_file_path="./gen_data/arc/results_annotation_llama3i_var_40_bs32.csv"

        # results_ori_path = "result/csqa/gen/phi3mini_csqa_gen_bs32/microsoft__Phi-3-mini-4k-instruct/samples_csqa_gen_2024-06-13T03-56-43.413176.json"
        # results_var_path = "result/csqa/gen/phi3mini_csqa_gen_40_bs32/microsoft__Phi-3-mini-4k-instruct/samples_csqa_gen_40_2024-06-13T04-17-16.051124.json"
        # output_file_path="./gen_data/csqa/results_annotation_phi3_var_40_bs32.csv"

        # results_ori_path = "result/truthfulqa/mc2/phi3mini_truthfulqa_mc2_bs64/microsoft__Phi-3-mini-4k-instruct/samples_truthfulqa_mc2_2024-06-12T23-27-21.894027.json"
        # results_var_path = "result/truthfulqa/mc2_sample_answer/phi3mini_truthfulqa_mc2_sample_answer_40_bs64/microsoft__Phi-3-mini-4k-instruct/samples_truthfulqa_mc2_sample_answer_40_2024-06-14T13-22-52.922931.jsonl"
        # output_file_path="./gen_data/truthfulqa/results_annotation_phi3_var_40_bs32.csv"
        results_ori = self._load_jsonl(results_ori_path)
        results_var = self._load_jsonl(results_var_path)
        count_round, count_err = 0, 0
        # # Open the input JSONL file and the output CSV file
        # with open(output_file_path, 'w', newline='') as csv_file:
        #     csv_writer = csv.writer(csv_file)
        #     # Write the header row to the CSV file
        #     csv_writer.writerow(['index', 'question', 'ground_truth_reasoning', 
        #             'ground_truth_answer', 'extracted_answer_from_model', 'complete_response_from_model'])
        #     # Read each line from the JSONL file
        for idx, row in enumerate(tqdm(results_var)):
            # ignore successful case
            # if row["acc"]: continue
            # verify if results_var matches results_ori in terms of index
            if results_ori[idx]["acc"] > row["acc"]:
                count_err += 1
            if results_ori[idx]["acc"] < row["acc"]:
                count_round += 1

        # print(f"{count_round} out of {len(results_ori)} cases are almost correct ...")
        print(f"{count_err} out of {len(results_ori)} cases are wrong ...")
        print(f"{count_round} out of {len(results_ori)} cases are corrected ...")


def main():
    args = parse_args()
    gen = GPTAnnotator(args)


    function = getattr(gen, args.task, "generate_annotation_set_for_code")
    if callable(function):
        function()
    else:
        raise ValueError("Choose a pre-defined task to execute ... ")

if __name__ == "__main__":
    main()
    


        
        
        

