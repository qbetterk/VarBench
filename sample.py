#!/usr/bin/env python3
#
import sys, os, pdb
import json, math, re, csv
import random
import signal
import pandas as pd
import numpy as np

from tqdm import tqdm
from base.base import BaseClass
from base.parse_args import parse_args
from base.utils import *

SPLIT="; "

class GPTSampler(BaseClass):
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
        

    def generate_test_set_gsm8k(self):
        """
        load problem statment template, sample new values and generate new statements"""
        self.train_data_path = "./data/grade-school-math/grade_school_math/data/train.jsonl"
        self.data_path = f"./gen_data/gsm8k/test_gsm8k_{self.model}.jsonl" if not self.args.data_path else self.args.data_path
        self.save_dir = f"./gen_data/gsm8k/sample_{self.random_seed}" if not self.args.save_dir else self.args.save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        self.save_path = os.path.join(self.save_dir, "test.jsonl")
        data = self._load_jsonl(self.data_path)
        data_var =  open(self.save_path, "a")
        count = 0
        for row in tqdm(data):
            if not row["question_delex"]: continue
            # sample new values for each variable
            # variables = {var:self.sample_value(value) for var, value in row["variables"].items()}
            variables = self.sample_value_from_range(row["index"], row["input_range"], row["variables"])
            if type(variables) == str: continue
            # print("Original Variables:", row["variables"])
            # print("Variables with newly sampled values:", variables)
            # insert values into statment template
            question = self.insert_value(variables, row["question_delex"])
            # print(f"***Problem statement with values: \n{question}")
            # compute answer based on solution code
            answer = run_func(variables, row["func"])
            answer = self.normalize(answer)
            if not is_number(answer): continue
            # print(f"***Computed answer: {answer}")
            row_new ={
                "question": question,
                "answer": f"#### {answer}",
                "index": row["index"],
            }
            data_var.write(json.dumps(row_new) + "\n")
            count += 1
        data_var.close()
        print(f"Successfully reconstruct {count} / {len(data)} problems from the original GSM8K")


    def normalize(self, number):
        # Convert string to float, format it with 3 decimal places, and strip trailing zeros
        formatted_number = f"{float(number):.7f}".rstrip('0').rstrip('.')
        # Combine the prefix and the formatted number
        return formatted_number


    def sample_value(self, value_dict, hardness=0, range=None):
        """
        deprecated
        modify variable values based on certain strategy & hardness
        currently, the sampling strategy is based on original value.
        if the original value is larger than 1, we consider it as 
        a normal integer and we sample value from value/2 to 2 * value
        if the original value is less than 1, we consider it as
        a fraction and we sample a decimal between 0 and 1"""
        # **** the first 67 cases does not have comments
        value = value_dict["value"] if type(value_dict) == dict else value_dict
        # TODO: use value_dict["comment"] to give a range for value
        if value >= 1: # integer
            left = -1 * int(value / 2)
            right = int(2 * value)
            value_sample = value + random.randint(left, right)
        elif value > 0:
            value_sample = float("%0.2f" % random.random())
        elif value < 0: # temperature degree
            right = -1 * int(value / 2)
            left = int(2 * value)
            value_sample = value + random.randint(left, right)
        else: # value == 0
            value_sample = value
        return value_sample


    def sample_value_from_range(self, idx, input_range, variables_ori):
        input_range = self.complete_range_wi_original_value(input_range, variables_ori)
        try:
            sample_values = generate_sample_values(input_range)
            return sample_values
        except Exception as err:
            print(f"ID: {idx}", str(err))
            return str(err)


    def complete_range_wi_original_value(self, input_range, variables_ori):
        # input_range = {key_:value_ for key_, value_ in input_range.items() if key_ in variables_ori}
        for key_, value_ in variables_ori.items():
            if key_ not in input_range:
                input_range[key_] = str(value_["value"])
        return input_range

    
    def insert_value(self, variables, question_delex):
        def replace_expression(match):
            # this is designed for cases where {} contains a expression rather than a variable
            expression = match.group(1)
            return str(eval(expression, variables.copy()))
        question_delex = re.sub(r'{([^}]+)}', replace_expression, question_delex)
        question_delex = question_delex.format(**variables)
        return question_delex


    def copy_train_set(self):
        train_data = self._load_jsonl(self.train_data_path)
        for idx, row in enumerate(tqdm(train_data)):
            row["index"] = idx
        self._save_jsonl(train_data, os.path.join(self.save_dir, "train.jsonl"))


    def train_and_test(self):
        self.generate_test_set_gsm8k()
        self.copy_train_set()


    def generate_dev_set_csqa(self, shuffle=False):
        """
        sample both positive and negative choices for csqa
        try to use arc's evaluation code, so convert csqa into arc format
        1. if the random seed is zero, then we generate the original data
        2. if candidate list is empty, then we use the original choices
        3. all choices are capitalized for the first letter"""
        self.data_path = "./gen_data/csqa/dev_csqa_gpt4o.jsonl" if not self.args.data_path else self.args.data_path
        shuffle = True
        if shuffle:
            self.save_dir = f"./gen_data/csqa/shuffle_{self.random_seed}" if not self.args.save_dir else self.args.save_dir
        else:
            self.save_dir = f"./gen_data/csqa/sample_{self.random_seed}" if not self.args.save_dir else self.args.save_dir
        os.makedirs(self.save_dir, exist_ok=True)

        data = self._load_jsonl(self.data_path)
        data_hf = []
        for idx, row in enumerate(data):
            answerKey = random.choice(['A', 'B', 'C', 'D', 'E']) if shuffle else row["answerKey"]
            row_hf = {
                "id": row["id"],
                "question": row["question"]["stem"] + " Answer Choices:",
                "choices" : {
                    "text": [],
                    "label": [],
                },
                "answerKey": answerKey,
                "index": idx,
            }
            posi_cands, nega_cands = [], []
            for choice in row["question"]["choices"]:
                if choice["label"] == row["answerKey"]:
                    posi_cands.append(choice["text"])
                else:
                    nega_cands.append(choice["text"])
            # sample from candidate list
            # if random seed == 0 or no candidates available, we use the original one
            if self.random_seed == 0 or shuffle:
                posi_choice = posi_cands
            else:
                posi_choice = random.sample(posi_cands + row["candidates"]["positive"], k=1)
            if len(nega_cands) == len(row["question"]["choices"])-1 or self.random_seed == 0 or shuffle:
                nega_choices = nega_cands
            else:
                nega_choices = random.sample(nega_cands + row["candidates"]["negative"], k=len(row["question"]["choices"])-1)
            # insert sampled values into choice list
            for choice in row["question"]["choices"]:
                if choice["label"] == row_hf["answerKey"]:
                    choice_text = posi_choice.pop().capitalize()
                else:
                    choice_text = nega_choices.pop().capitalize()
                row_hf["choices"]["text"].append(choice_text)
                choice_label = f"({choice['label'].lower()})"
                row_hf["choices"]["label"].append(choice_label)
                row_hf["question"] += f" {choice_label} {choice_text}"
            row_hf["answerKey"] = f"({row_hf['answerKey'].lower()})"
            data_hf.append(row_hf)
        self._save_jsonl(data_hf, os.path.join(self.save_dir, "validation.jsonl"))


    def generate_test_set_arc(self):
        """
        for arc_challenge
        load problem statment template, sample new values and generate new statements"""

        self.data_path = f"./gen_data/arc/challenge/test_arc_challenge_gpt4o.jsonl" if not self.args.data_path else self.args.data_path
        self.save_dir = f"./gen_data/arc/challenge/sample_{self.random_seed}" if not self.args.save_dir else self.args.save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        self.train_data_path = "data/ARC-V1-Feb2018-2/ARC-Challenge/ARC-Challenge-Train.jsonl"
        self.val_data_path = "data/ARC-V1-Feb2018-2/ARC-Challenge/ARC-Challenge-Dev.jsonl"

        data = self._load_jsonl(self.data_path)
        # data_new_value = []
        for row in tqdm(data):
            # sample new values for positive variable
            if "candidates" not in row: pdb.set_trace()
            posi_list, nega_list = row["candidates"]["positive"], row["candidates"]["negative"]
            for choice in row["question"]["choices"]:
                if choice["label"] == row["answerKey"]:
                    posi_list.append(choice["text"])
                else:
                    nega_list.append(choice["text"])
            # print(row, "\n")
            posi_choice = random.sample(posi_list, k=1)
            nega_choices = random.sample(nega_list, k=len(row["question"]["choices"])-1)
            for choice in row["question"]["choices"]:
                if choice["label"] == row["answerKey"]:
                    choice["text"] = posi_choice.pop()
                else:
                    choice["text"] = nega_choices.pop()

        data_hf = self.convert_arc_hf(data)
        self._save_jsonl(data_hf, os.path.join(self.save_dir, "test.jsonl"))
        
        train_data = self._load_jsonl(self.train_data_path)
        train_data_hf = self.convert_arc_hf(train_data)
        self._save_jsonl(train_data_hf, os.path.join(self.save_dir, "train.jsonl"))

        val_data = self._load_jsonl(self.val_data_path)
        val_data_hf = self.convert_arc_hf(val_data)
        self._save_jsonl(val_data_hf, os.path.join(self.save_dir, "validation.jsonl"))  


    def convert_arc_hf(self, data_ori):
        """
        this function convert the original arc data format into huggingface format
        so that lm-eval can handle our processed data
        original format:
        {
            "id": "Mercury_7111125", 
            "question": {
                "stem": question, 
                "choices": [
                    {"text": "temperature increases during an infection", "label": "A"}, 
                    ... (up to five choices)
                ],
                "choices_ori": []
            }, 
            "answerKey": "B", 
            "index": 1168, 
            "answerKey_ori": "A"
        }
        hf format:
        {
            "id": "Mercury_7111125", 
            "question": question, 
            "choices": {
                "text": ["dry palms", ...],
                "label": ["A", "B", "C", "D"],
            "answerKey": "B", 
            "index": 1168, 
            "answerKey_ori": "A",
            "choices_ori": [],
        """
        data_hf = []
        for idx, row_ori in enumerate(data_ori):
            row_hf = {
                "id": row_ori["id"],
                "question": row_ori["question"]["stem"],
                "choices" : {
                    "text": [],
                    "label": [],
                },
                "answerKey": row_ori["answerKey"],
                "index": idx,
                "answerKey_ori": "",
                "choices_ori": [],
            }
            for choice in row_ori["question"]["choices"]:
                row_hf["choices"]["text"].append(choice["text"])
                row_hf["choices"]["label"].append(choice["label"])
            # row_hf["choices_ori"] = row_ori["question"].get("choices_ori", [])
            # row_hf["answerKey_ori"] = row_ori.get("answerKey_ori", "")
            data_hf.append(row_hf)
        return data_hf


    def generate_val_set_truthfulqa(self, new_question=True):
        """
        for truthfulqa
        load problem statment template, sample new values and generate new statements"""

        self.data_path = f"./gen_data/truthfulqa/validation_truthfulqa_gpt4o.jsonl" if not self.args.data_path else self.args.data_path
        if new_question:
            self.save_dir = f"./gen_data/truthfulqa/sample_both_{self.random_seed}" if not self.args.save_dir else self.args.save_dir
        else:
            self.save_dir = f"./gen_data/truthfulqa/sample_answer_{self.random_seed}" if not self.args.save_dir else self.args.save_dir
        os.makedirs(self.save_dir, exist_ok=True)

        data = self._load_jsonl(self.data_path)
        data_new = []
        for row in tqdm(data):
            # sample new values for positive variable
            if "candidates" not in row: pdb.set_trace()
            if len(row['candidates']) < 2:
                data_new.append({
                    'question' : question,
                    'mc1_targets' : { # unchanged
                        'choices' : list(row['mc1_targets'].keys()),
                        'labels'  : list(row['mc1_targets'].values()),
                    },
                    'mc2_targets' : {
                        'choices' : list(row['mc2_targets'].keys()),
                        'labels'  : list(row['mc1_targets'].values()),
                    },
                    'index' : row["index"],
                })
                continue
            if row['question'] != row['candidates'][0]['question']: pdb.set_trace()
            if not new_question:
                question = row['candidates'][0]['question']
                posi_list, nega_list = row["candidates"][0]["positive"], row["candidates"][0]["negative"]
                posi_num, nega_num = 0, 0
                for answer, label in row["mc2_targets"].items():
                    if label:
                        posi_list.append(answer)
                        posi_num += 1
                    else:
                        nega_list.append(answer)
                        nega_num += 1
            else:
                sample_num = random.choice(range(len(row["candidates"])-1)) + 1
                question = row["candidates"][sample_num]["question"]
                posi_list = row["candidates"][sample_num]["positive"]
                nega_list = row["candidates"][sample_num]["negative"]
                posi_num_ori = sum(row["mc2_targets"].values())
                nega_num_ori = len(row["mc2_targets"]) - posi_num_ori
                posi_num = min(len(posi_list), posi_num_ori)
                nega_num = min(len(nega_list), nega_num_ori)
                
            posi_choices = random.sample(posi_list, k=posi_num)
            nega_choices = random.sample(nega_list, k=nega_num)
            data_new.append({
                'question' : question,
                'mc1_targets' : { # unchanged
                    'choices' : list(row['mc1_targets'].keys()),
                    'labels'  : list(row['mc1_targets'].values()),
                },
                'mc2_targets' : {
                    'choices' : posi_choices + nega_choices,
                    'labels'  : [1 for _ in posi_choices] + [0 for _ in nega_choices],
                },
                'index' : row["index"],
            })

        self._save_json(data_new, os.path.join(self.save_dir, "validation.json"))

def main():
    args = parse_args()
    gen = GPTSampler(args)
    # gen.generate_test_set()


    function = getattr(gen, args.task, "generate_test_set_gsm8k")
    if callable(function):
        function()
    else:
        raise ValueError("Choose a pre-defined task to execute ... ")

if __name__ == "__main__":
    main()
    


        
        
        

