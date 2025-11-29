import os
import re
import pdb
import torch
from peft import get_peft_model, prepare_model_for_kbit_training
from transformers import (
   AutoTokenizer, 
   AutoModelForCausalLM, 
   DataCollatorForSeq2Seq,
)

from data import load_infer_dataset, tokenize_dataset, load_jsonl
from config import LoRAConfig, bnbConfig
from train import train_and_save_model



def LoRA(model):
   model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
   model = get_peft_model(model, LoRAConfig)  
   return model


def compute_accuracy(
      model, 
      infer_path,
      tokenizer,
   ):

   valid_data_path = os.path.join(infer_path, 'infer_res_valid.jsonl')
   
   # load the best model
   device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
   model.to(device)
   
   # load the label data
   valid_data = load_jsonl(valid_data_path)

   # compute the score
   model.eval()
   correct = 0
   total = 0
   for line in valid_data:

      inputs = tokenizer(
         line["input"],
         return_tensors='pt'
      ).to(device)

      with torch.no_grad():
         output = model.generate(
            **inputs,
            temperature = 0.4,
            do_sample = True,
            pad_token_id=tokenizer.pad_token_id,
         )
      
      res = tokenizer.decode(
         output[0],
         skip_special_tokens = True,
      )

      model_ans = re.search(r'정답:\s*(\d+)', res)
      model_ans = int(model_ans.group(1)) if model_ans else None

      if model_ans == line["answer"]:
         correct += 1
      total += 1

   print("[Accuracy : {:.4f}]".format(correct/total))
   return 



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
         data_collator,
         save_path,  
         model_name,
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
      infer_path,
      tokenizer,
   )


   

if __name__ == "__main__":
   model_name = "kakaocorp/kanana-1.5-8b-instruct-2505"
   mode_flag = 'vanilla'       # or 'train'

   main(model_name, mode_flag)

   # Models
   # kakaocorp/kanana-1.5-8b-instruct-2505
   # LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct
   # K-intelligence/Midm-2.0-Base-Instruct