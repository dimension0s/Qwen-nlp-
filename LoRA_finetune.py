# 使用LoRA微调

import json,torch,os
from torch.utils.data import DataLoader,Dataset
from transformers import AutoModelForCausalLM,AutoTokenizer
from transformers import Trainer,DataCollatorForSeq2Seq,TrainingArguments
from Qwenfinetuning.device import device
from Qwenfinetuning.model import model


# loRA微调
from peft import LoraConfig,TaskType,get_peft_model

lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,  # 根据你的任务选择适当的类型
    # target_modules=['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj'],
    # inference_mode=False,  # 训练模式
    r=8,  # loRA秩
    lora_alpha=32,  # LoRA的缩放因子
    lora_dropout=0.1  # dropout比例
)
model = get_peft_model(model, lora_config)



