import os
import json
import pandas as pd
import pdb

from typing import Callable
from datasets import Dataset as HFDataset
from datasets import load_dataset as hf_load_dataset


INPUT = '''
{passage}                                          

{problem}
{options}

한두 문장으로 매우 간단한 해설을 작성한 뒤, 정답 번호 하나를 제시하라.

출력 형식
"
[해설]
(한두 문장의 간단한 해설)

정답: (1~5번 중 무조건 정답 하나)
'''


def load_infer_dataset(path: str, subpath)-> HFDataset:

    data_file = os.path.join(path, subpath)
    dataset = hf_load_dataset("json", data_files = data_file, split="train")

    return dataset


def load_probs_dataset(path: str)-> HFDataset:

    data_files = {
        "train" : "train.jsonl",
        "validation" : "validation.jsonl",
        "test" : "test.jsonl"
    }

    for k,v in data_files.items():
        data_files[k] = os.path.join(path, v)
        
    dataset = hf_load_dataset("json", data_files = data_files)

    return dataset


def tokenize_dataset(
        split: HFDataset,
        tokenizer: Callable,
):
    def process_func(unit):
        if dstype == 'i':
            
            input_text = unit["input"]
            label_text = unit["label"]

            full_text = input_text + " " + label_text

            full_toks = tokenizer(full_text)
            input_toks = tokenizer(input_text)

            input_len = len(input_toks["input_ids"])

            labels = full_toks["input_ids"].copy()
            labels[:input_len] = [-100] * input_len

            return dict(
                input_ids=full_toks["input_ids"],
                attention_mask=full_toks["attention_mask"], 
                labels=labels
            )

    ds = split.map(process_func, batched=False)
    return ds
    
    


def Logging(output, save_path):
    """
    Old version of logging, it is not used.
    """

    if not os.path.exists(save_path):
        os.mkdir(save_path)

    res_path = os.path.join(save_path, 'output.xlsx')

    res = pd.DataFrame(output)
    res.to_excel(res_path)



def rearrange_data(src_data_path, res_data_path):
    '''
    Being used only if it is necessary (No need to care).

    The function for changing the file format from json to jsonl.
    Original json file has the form of too much nested dictionaries. 
    we used a rearrange_data function to process this problem.
    '''

    if not os.path.exists(res_data_path):
        folder = os.path.dirname(res_data_path)
        os.makedirs(folder)

    with open(src_data_path, "r", encoding='utf-8') as f:
        loaded_data = json.load(f)

    with open(res_data_path, "w", encoding='utf-8') as f:
        for _, v1 in loaded_data.items():
            for _, v2 in v1[0]["paragraph_groups"].items():
                json.dump(v2, f, ensure_ascii=False)
                f.write("\n")
