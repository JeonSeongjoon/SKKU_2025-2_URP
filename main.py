import torch
import pandas as pd
from peft import get_peft_model, prepare_model_for_kbit_training

from data import data_load, toExcel
from model import KoLLM
from config import LoRAConfig, bnbConfig


def test_KoLLM():
   data = data_load(path)

   LLM = KoLLM(model_name, bnbConfig)
   LLM.model = LoRA(LLM.model)
   
   outputs = LLM.Inference(data)
   toExcel(outputs)
   
   return


def LoRA(model):
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)
    model = get_peft_model(model, LoRAConfig)  
    return model



if __name__ == "__main__":
   model_name = "EleutherAI/polyglot-ko-5.8b"
   path = './data/dataset.json'

   test_KoLLM()