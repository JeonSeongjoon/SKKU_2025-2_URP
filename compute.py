import re
import os
import json
import torch

from torch.utils.data import DataLoader

from peft import PeftModel
from transformers import AutoModelForCausalLM
from config import bnbConfig
from main import LoRA


def compute_accuracy(
      best_model_pth,
      test_ds, 
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
   )

   if best_model_pth:
      model = LoRA(model)
   else:
      model = PeftModel.from_pretrained(model, best_model_pth)

   # set the model info
   model_li = model_name.split('/')
   model_info = '-'.join(model_li)
   
   # load the best model
   device = next(model.parameters()).device
   
   # load the label data
   labels = test_ds["answer"]
   test_ds = test_ds.remove_columns(["label", "labels", "input", "answer"])  
   test_ds.set_format("torch")

   dataloader = DataLoader(
       test_ds,
       shuffle = False,
       batch_size = 1,                                         
   )

   print("Evaluating the performance of the model.")

   # compute the score
   record = []

   model.eval()
   correct = 0
   total = 0

   for batch in dataloader:
      input = {k: v.to(device) for k, v in batch.items()}

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
        "answer" : labels[total] 
      })

     
      # accuracy
      model_ans = re.search(r'정답:\s*(\d+)', res)
      model_ans = int(model_ans.group(1)) if model_ans else None
     
      if model_ans == labels[total]:
         correct += 1

      total += 1


   with open(os.path.join(f'output_{model_info}_({mode_flag}).jsonl'), "w", encoding='utf-8') as f:
        for line in record:
            json.dump(line, f, ensure_ascii=False)
            f.write("\n")

   print("[model: {}_({})]  Accuracy: {:.4f}".format(model_info, mode_flag, correct/total))
   return 
