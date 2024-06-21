#!/usr/bin/env python3
#
import sys, os, pdb
import json, random
import numpy as np
import pandas as pd
from tqdm import tqdm

random.seed(42)
SPLIT="; "

class BaseClass(object):
    def __init__(self, args) -> None:
        self.args = args
        self.random_seed = args.seed
        self._decide_model(args)
        if args.seed:
            self._set_seed(args.seed)

    def _set_seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)

    def _decide_model(self, args):
        #"gpt-4-1106-preview", # "gpt-3.5-turbo", # "gpt-4", "gpt-3.5-turbo-1106"
        if args.model_name_or_path == "gpt-4":
            self.model="gpt4"
        elif args.model_name_or_path == "gpt-4-turbo":
            self.model="gpt4t"
        elif args.model_name_or_path == "gpt-3.5-turbo":
            self.model="gpt35t"
        elif args.model_name_or_path == "gpt-3.5-turbo-0125":
            self.model="gpt35tnew"
        elif args.model_name_or_path == "gpt-4o":
            self.model = "gpt4o"
        elif args.model_name_or_path.startswith("gpt-4"):
            self.model = "gpt4t"
        elif args.model_name_or_path.startswith("gpt-3.5"):
            self.model = "gpt35t"
        else:
            self.model = "gemini"


    def _load_json(self, path):
        if path is None or not os.path.exists(path):
            raise IOError(f"File doe snot exists: {path}")
        print(f"Loading data from {path} ...")
        with open(path) as df:
            data = json.loads(df.read())
        return data


    def _load_jsonl(self, path):
        if path is None or not os.path.exists(path):
            raise IOError(f"File doe snot exists: {path}")
        print(f"Loading data from {path} ...")
        data = []
        with open(path) as df:
            for row in df.readlines():
                data.append(json.loads(row))
        return data
    

    def _load_txt(self, path=None, split_tok="\n", readlines=True):
        if path is None or not os.path.exists(path):
            raise IOError('File does not exist: %s' % path)
        with open(path) as df:
            if readlines:
                data = df.read().strip().split(split_tok)
            else:
                data = df.read()
        return data


    def _load_csv(self, path=None, sep=","):
        if path is None or not os.path.exists(path):
            raise IOError('File does not exist: %s' % path)
        with open(path) as df:
            data = pd.read_csv(df, sep=sep)
        return data


    def _save_json(self, data, file_path):
        dir_path = "/".join(file_path.split("/")[:-1])
        if dir_path and not os.path.exists(dir_path):
            os.makedirs(dir_path)
        print(f"Saving file {file_path} ...")
        with open(file_path, "w") as tf:
            json.dump(data, tf, indent=2)


    def _save_jsonl(self, data, file_path):
        dir_path = "/".join(file_path.split("/")[:-1])
        if dir_path and not os.path.exists(dir_path):
            os.makedirs(dir_path)
        print(f"Saving file {file_path} ...")
        with open(file_path, mode='w') as file:
            for row in data:
                json.dump(row, file)
                file.write('\n')