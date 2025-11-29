import os
import pdb
import torch
from peft import get_peft_model, prepare_model_for_kbit_training
from transformers import (
   AutoTokenizer, 
   AutoModelForCausalLM, 
   DataCollatorForSeq2Seq,
)

from data import load_infer_dataset, tokenize_dataset
from config import LoRAConfig, bnbConfig
from train import train_and_save_model
from compute import compute_accuracy



def LoRA(model):
   model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
   model = get_peft_model(model, LoRAConfig)  
   return model


def main(model_name, mode_flag):

   infer_path = './data/infer_result'
   save_path = './result'
   
   # load dataset
   train_ds = load_infer_dataset(infer_path, 'infer_res_train.jsonl')
   test_ds = load_infer_dataset(infer_path, 'infer_res_valid.jsonl')

   # preprocess dataset
   tokenizer = AutoTokenizer.from_pretrained(model_name)
   train_ds = tokenize_dataset(train_ds, tokenizer)
   test_ds = tokenize_dataset(test_ds, tokenizer)
   
   
   model = AutoModelForCausalLM.from_pretrained(
      model_name,
      quantization_config = bnbConfig,
      device_map="auto",
      torch_dtype=torch.float16,
      trust_remote_code=True,
   )
   model = LoRA(model)


   if mode_flag == 'train':

      data_collator = DataCollatorForSeq2Seq(
         tokenizer=tokenizer,
         model=model,
         padding=True,
         label_pad_token_id=-100  # loss 계산 시 패딩 토큰 무시
      )

      # train model
      best_model_pth = train_and_save_model(
         model,
         train_ds,
         test_ds,
         save_path,  
         model_name,
         mode_flag,              #change
      )

      # reinitialize the model
      model = AutoModelForCausalLM.from_pretrained(
         model_name,
         quantization_config = bnbConfig,
         device_map="auto",
         torch_dtype=torch.float16,
      )
      model = LoRA(model)
      model.load_state_dict(torch.load(best_model_pth, map_location='cpu'), strict=False)


   # compute the score
   compute_accuracy(
      model,
      test_ds, 
      tokenizer,
      model_name,
      mode_flag,
   )


   

if __name__ == "__main__":
   model_name = "kakaocorp/kanana-1.5-8b-instruct-2505"
   mode_flag = 'vanilla'       # or 'train'

   main(model_name, mode_flag)

   # Models
   # kakaocorp/kanana-1.5-8b-instruct-2505
   # LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct
   # K-intelligence/Midm-2.0-Base-Instruct