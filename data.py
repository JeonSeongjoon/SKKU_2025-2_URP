import os
import json
import pandas as pd
import pdb

from typing import Callable
from datasets import Dataset as HFDataset
from datasets import load_dataset as hf_load_dataset


INPUT = '''
다음 지문을 읽고 문제에 대한 답을 작성하라

{paragraph}                                          

{problem}
{options}

출력 형식은 아래와 같다.
"
[해설]
(한두 문장의 간단한 해설)

정답: (1~5번 중 무조건 정답 하나)
"
[해설] :
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
      
      input_text = unit["input"]
      label_text = unit["label"]

      full_text = input_text + " " + label_text

      full_toks = tokenizer(
        full_text,
        #max_length = 2,
        #truncation = True
      )
      input_toks = tokenizer(
        input_text,
        #max_length = 2,
        #truncation = True
      )

      input_len = len(input_toks["input_ids"])
      #pdb.set_trace()


      labels = full_toks["input_ids"].copy()
      labels[:input_len] = [-100] * input_len

      return dict(
          input_ids=full_toks["input_ids"],
          attention_mask=full_toks["attention_mask"], 
          labels=labels
      )

    ds = split.map(process_func, batched=False)
    return ds



def load_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():  # 빈 줄 건너뛰기
                data.append(json.loads(line))
    return data



def setting_problem_form(paragraph, problem, options):

    input_text = INPUT.format(
        paragraph=paragraph,
        problem=problem,
        options=options
    )

    return input_text
