import re
import os
import json
import torch

from torch.utils.data import DataLoader

from peft import PeftModel, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM
from config import bnbConfig, LoRAConfig
from data import load_jsonl, setting_problem_form


def LoRA(model):
   model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
   model = get_peft_model(model, LoRAConfig)  
   return model

def compute_accuracy(
      best_model_pth, 
      tokenizer,
      model_name,
      mode_flag,
   ):

   # reinitialize the model
   model = AutoModelForCausalLM.from_pretrained(
      model_name,
      quantization_config = bnbConfig,
      device_map="auto",
      torch_dtype=torch.float16,
      trust_remote_code=True,
   )

   if best_model_pth:
      model = PeftModel.from_pretrained(model, best_model_pth)
   else:
      model = LoRA(model)

   # set the model info
   model_li = model_name.split('/')
   model_info = '-'.join(model_li)
   
   # load the best model
   device = next(model.parameters()).device
   
   # load the label data
   test_ds_path = './data/KSAT_LEET_probs/validation.jsonl'
   test_ds = load_jsonl(test_ds_path)
  

   print("Evaluating the performance of the model.")

   # compute the score
   record = []

   model.eval()
   correct = 0
   total = 0

   for prob in test_ds:
      
      input_text = setting_problem_form(prob["paragraph"], prob["problem"], prob["options"])

      input = tokenizer(input_text, return_tensors='pt')

      with torch.no_grad():
         output = model.generate(
            **input,
            temperature = 0.4,
            do_sample = True,
         )
      
      res = tokenizer.decode(
         output[0],
         skip_special_tokens = True,
      )


      # record the output
      record.append({
        "output" : res, 
        "answer" : prob["answer"] 
      })

     
      # accuracy
      model_ans = re.search(r'정답:\s*(\d+)', res)
      model_ans = int(model_ans.group(1)) if model_ans else None
     
      if model_ans == prob["answer"]:
         correct += 1

      total += 1


   with open(os.path.join(f'output_{model_info}_({mode_flag}).jsonl'), "w", encoding='utf-8') as f:
        for line in record:
            json.dump(line, f, ensure_ascii=False)
            f.write("\n")

   print("[model: {}_({})]  Accuracy: {:.4f}".format(model_info, mode_flag, correct/total))
   return 
