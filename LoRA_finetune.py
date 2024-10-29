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
# 如果出现问题：使用的 CUDA 版本或 PyTorch 不支持在 BFloat16 下执行某些操作：
# 将模型的输入和参数转换为 Float16 或 Float32
# model.half()  # 将模型转换为 float16
# 或者：使用 float32
# model.float()

