# 调用模型
from transformers import AutoTokenizer,AutoModelForCausalLM
from Qwen微调nlp任务.device import device

model_name = "E:/NLP任务/离线模型/Qwen2-1.5b"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name,torch_dtype="auto")
model = model.to(device)