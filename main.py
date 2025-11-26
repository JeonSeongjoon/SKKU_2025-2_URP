import os
import pdb
import torch
from peft import get_peft_model, prepare_model_for_kbit_training
from transformers import (
   AutoTokenizer, 
   AutoModelForCausalLM, 
   DataCollatorWithPadding,
)

from data import load_infer_dataset, load_probs_dataset, tokenize_dataset
from config import LoRAConfig, bnbConfig, getConfig
from train import train_and_save_model



def LoRA(model):
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)
    model = get_peft_model(model, LoRAConfig)  
    return model



def main(model_name, data_info):
  
   infer_path = './data/infer_result'
   save_path = './result'
   
   # load dataset
   train_ds = load_infer_dataset(infer_path, 'train_res.jsonl')
   test_ds = load_infer_dataset(infer_path, 'train_res_valid.jsonl')

   # preprocess dataset
   tokenizer = AutoTokenizer.from_pretrained(model_name)
   train_ds = tokenize_dataset(train_ds, tokenizer, 'i')
   test_ds = tokenize_dataset(test_ds, tokenizer, 'i')
   
   
   model = AutoModelForCausalLM.from_pretrained(
      model_name,
      quantization_config = bnbConfig,
      device_map="auto",
      torch_dtype=torch.float16,
   )
   model = LoRA(model)

   data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
   

   # train model
   train_and_save_model(
      model,
      train_ds,
      test_ds,
      data_collator,
      save_path,  
      model_name,
   )
   

if __name__ == "__main__":
   model_name = "MLP-KTLim/llama-3-Korean-Bllossom-8B"
   data_info = "infer_result"                          #KSAT_LEET_probs

   main(model_name, data_info)