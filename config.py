from transformers import BitsAndBytesConfig, TrainingArguments
from peft import LoraConfig
import torch


# 8bits quantization config
bnbConfig = BitsAndBytesConfig(
    load_in_8bit=True,
    bnb_8bit_compute_dtype=torch.float16,
    bnb_8bit_quant_type="nf4",
    bnb_8bit_use_double_quant=True,
)


# LoRA config
LoRAConfig = LoraConfig(
    r=16,                    # rank
    lora_alpha=32,           # alpha
    target_modules = ["q_proj", "k_proj" "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,       # dropout rate
    bias="none",             # learning bias
    task_type="CAUSAL_LM"   
)
