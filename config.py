from transformers import BitsAndBytesConfig, TrainingArguments
from peft import LoraConfig
import torch


MODEL_CONFIG = {
    
    "basic" : {
        "lr" : 1e-4,
        "epochs" : 5,
        "batch_size": 1,
    },
}


def getConfig(model_name):

    if model_name in MODEL_CONFIG.keys():
        modelConfig = MODEL_CONFIG[model_name]
    else:
        modelConfig = MODEL_CONFIG["basic"]

    return modelConfig



# 8bits quantization config
bnbConfig = BitsAndBytesConfig(
    load_in_8bit=True,
    bnb_8bit_compute_dtype=torch.float16,
)


# LoRA config
LoRAConfig = LoraConfig(
    r=16,                    # rank
    lora_alpha=32,           # alpha
    target_modules = "all-linear",
    lora_dropout=0.05,       # dropout rate
    bias="none",             # learning bias
    task_type="CAUSAL_LM"   
)
