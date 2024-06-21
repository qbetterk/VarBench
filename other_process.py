#!/usr/bin/env python3
#
import sys, os, pdb
import json, math, re
import random

from tqdm import tqdm
from base.api import openai_api_chat, gemini_api_complete
from base.base import BaseClass
from base.parse_args import parse_args

SPLIT="; "

class GPTGenerator(BaseClass):
    def __init__(self, args) -> None:
        super().__init__(args)
        self.args = args
        self.random_seed = args.seed
        
        self.prompt_sys_path = "./prompt/prompt_paraphrase.txt"
        if not args.save_dir: raise ValueError("Invalid saving directory ... ")
        save_filename = args.save_filename if args.save_filename else "test.jsonl"
        self.save_path = os.path.join(args.save_dir, save_filename)
        self.save_dir = args.save_dir
        if self.save_dir: os.makedirs(self.save_dir, exist_ok=True)


    def _load_data(self, data_name="gsm8k"):
        self.train_data_path = ""
        self.val_data_path = ""
        if data_name == "gsm8k":
            data_path = "./data/grade-school-math/grade_school_math/data/test.jsonl"
            self.train_data_path = "./data/grade-school-math/grade_school_math/data/train.jsonl"
            return self._load_jsonl(data_path)
        elif data_name == "commonsenseqa":
            data_path = "./data/commonsenseqa/dev_rand_split.jsonl"
            return self._load_jsonl(data_path)
        elif data_name == "arc_easy":
            data_path = "./data/ARC-V1-Feb2018-2/ARC-Easy/ARC-Easy-Test.jsonl"
            self.train_data_path = "data/ARC-V1-Feb2018-2/ARC-Easy/ARC-Easy-Train.jsonl"
            self.val_data_path = "data/ARC-V1-Feb2018-2/ARC-Easy/ARC-Easy-Dev.jsonl"
            return self._load_jsonl(data_path)
        elif data_name == "arc_challenge":
            data_path = "./data/ARC-V1-Feb2018-2/ARC-Challenge/ARC-Challenge-Test.jsonl"
            self.train_data_path = "data/ARC-V1-Feb2018-2/ARC-Challenge/ARC-Challenge-Train.jsonl"
            self.val_data_path = "data/ARC-V1-Feb2018-2/ARC-Challenge/ARC-Challenge-Dev.jsonl"
            return self._load_jsonl(data_path)
        elif data_name == "truthfulqa":
            data_path = "./data/truthfulqa/data/mc_task.json"
            return self._load_json(data_path)
        else:
            raise ValueError("No matched dataset found ... ")


    def paraphrase_gsm8k(self):
        """
        paraphrase the problem statement while the original meaning
        prompt api to solve the new problem and verify if this answer 
        match the original one
        """
        SYS_PROMPT = self._load_txt(self.prompt_sys_path, readlines=False)
        data = self._load_data("gsm8k")
        data_done = self._load_jsonl(self.save_path) if os.path.exists(self.save_path) else []
        self.data_var = open(self.save_path, "a")

        for idx, row in enumerate(tqdm(data)):
            # resume and check already-generated rows
            if idx < len(data_done) and \
               row["question"] == data_done[idx]["question_ori"]:
                continue
            # paraphrase
            question = row["question"]
            output = openai_api_chat(self.args, input_seq=question, system_prompt=SYS_PROMPT, temperature=1)
            question_new = output.strip("### Rewritten Problem:\n")
            # TODO: prompt to compute new answer
            # saving new question
            answer_gt = extract_answer(row["answer"])
            row_var = self.set_row_var(question_new, row["question"], row["answer"], answer_gt, idx)
            self.data_var.write(json.dumps(row_var) + "\n")

        self.data_var.close()
        self.copy_train_val_set(extra_keys=["question_ori", "answer_num"])


    def copy_train_val_set(self, extra_keys):
        if self.train_data_path:
            train_data = self._load_jsonl(self.train_data_path)
            for idx, row in enumerate(tqdm(train_data)):
                row["index"] = idx
                for key_ in extra_keys:
                    row[key_] = ""
            self._save_jsonl(train_data, os.path.join(self.save_dir, "train.jsonl"))

        if self.val_data_path:
            val_data = self._load_jsonl(self.val_data_path)
            for idx, row in enumerate(tqdm(val_data)):
                row["index"] = idx
                for key_ in extra_keys:
                    row[key_] = ""
            self._save_jsonl(val_data, os.path.join(self.save_dir, "validation.jsonl"))


    def set_row_var(self, question, question_ori, answer, answer_num, index):
        return {
            "question"  : question,
            "question_ori": question_ori,
            "answer"    : answer,
            "answer_num": answer_num, # answer without reasoning
            "index"     : index,
        }


    def shuffle_arc(self):
        data = self._load_data(data_name="arc_challenge")
        for idx, row in enumerate(tqdm(data)):
            row["index"] = idx
            row['answerKey_ori'] = row['answerKey']
            choices, oracle = [], ""
            for choice in row['question']['choices']:
                if choice['label'] == row['answerKey']:
                    oracle = choice['text']
                choices.append(choice['text'])
            # shuffle
            random.shuffle(choices)
            # form new choices
            choice_label = ["A", "B", "C", "D", "E"]
            row['question']['choices_ori'] = row['question']['choices'][:]
            row['question']['choices'] = []
            for choice in choices:
                if not choice_label: pdb.set_trace()
                label = choice_label.pop(0)
                row['question']['choices'].append({
                    "text": choice,
                    "label": label,
                })
                if choice == oracle:
                    row['answerKey'] = label
        data_hf = self.convert_arc_hf(data)
        self._save_jsonl(data_hf, self.save_path)
        
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


    def shuffle_csqa(self):
        """
        this functino shuffle the order of answers in csqa"""
        pass


    def shuffle_truthfulqa(self):
        """
        this function shuffle the order of answers in mc2_targets"""
        data = self._load_data("truthfulqa")
        for idx, row in enumerate(tqdm(data)):
            row['index'] = idx
            if self.random_seed == 0:
                row['mc1_targets'] = self.shuffle_dict(row['mc1_targets'], no_shuffle=True)
                row['mc2_targets'] = self.shuffle_dict(row['mc2_targets'], no_shuffle = True)
            else: 
                row['mc1_targets'] = self.shuffle_dict(row['mc1_targets'])
                row['mc2_targets'] = self.shuffle_dict(row['mc2_targets'])
        self._save_json(data, self.save_path)


    def shuffle_dict(self, dict_ori, no_shuffle=False):
        # shuffle
        items = list(dict_ori.items())  # Convert dictionary items to a list
        if not no_shuffle:
            random.shuffle(items)             # Shuffle the list of items
        # return in format of huggingface
        keys = [key for key, value in items]   # Extract keys
        values = [value for key, value in items]  # Extract values
        return {"choices": keys, "labels": values}


def main():
    args = parse_args()
    gen = GPTGenerator(args)
    # gen.generate_code()

    function = getattr(gen, args.task, 'paraphrase_gsm8k')
    if callable(function):
        function()
    else:
        raise ValueError("Choose a pre-defined task to execute ... ")
    

if __name__ == "__main__":
    main()