import torch
import pandas as pd
from peft import get_peft_model, prepare_model_for_kbit_training

from data import data_load, Logging
from model import KoLLM
from config import LoRAConfig, bnbConfig


def main():
   data = data_load(data_path)

   LLM = KoLLM(model_name, bnbConfig)
   LLM.model = LoRA(LLM.model)
   
   outputs = LLM.Inference(data)
   Logging(outputs, save_path)
   
   return


def LoRA(model):
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)
    model = get_peft_model(model, LoRAConfig)  
    return model



if __name__ == "__main__":
   model_name = "MLP-KTLim/llama-3-Korean-Bllossom-8B"
   data_path = './data/dataset.json'
   save_path = './result'

   main()