# Qwen2-1.5b模型说明：该模型属于自回归生成模型，即Causal Language Model
# 因此没有解码器输入ID，这对数据处理有影响
# 先把代码放上去，初步问题是内存爆了： CUDA out of memory. Tried to allocate 36.00 MiB. GPU
# 后期主要是调参的工作


# 1.数据集处理
# 1.1)加载数据集
import torch,os,random,numpy
from transformers import AutoModelForCausalLM,AutoTokenizer
from torch.utils.data import Dataset,DataLoader

class LCSTS(Dataset):
    def __init__(self, data_file):
        self.data = self.load_data(data_file)

    def load_data(self, data_file):
        Data = {}
        with open(data_file, 'rt', encoding='utf-8') as f:
            for idx, line in enumerate(f):
                items = line.strip().split('!=!')
                assert len(items) == 2
                Data[idx] = {
                    'title': items[0],
                    'content': items[1],
                }
        return Data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


train_data = LCSTS("E:\\NLP任务\\生成式任务\\data\\lcsts_tsv\\data1_cutted.txt")
valid_data = LCSTS("E:\\NLP任务\\生成式任务\\data\\lcsts_tsv\\data2.txt")
test_data = LCSTS("E:\\NLP任务\\生成式任务\\data\\lcsts_tsv\\data3.txt")

# 打印测试，略

# device设置
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print(f"device num:{torch.cuda.device_count()}")
    print(f"device name:{torch.cuda.get_device_name()}")
else:
    device = torch.device("cpu")
    print("No GPU available,using the CPU instead.")



# 1.2）调用模型
model_name = "E:/NLP任务/离线模型/Qwen2-1.5b"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name,torch_dtype="auto")
model = model.to(device)

from torch.nn.utils.rnn import pad_sequence
# 1.3)分批处理
max_length = 128

def collote_fn(batch_samples):
    batch_inputs, batch_targets = [],[]
    for sample in batch_samples:
        text = sample['content']
        # 注意以下提示的设计，它可能是导致输入输出batch_size形状不匹配的主要原因
        messages = [
            {"role": "system", "content": "你是一个文本摘要的专家, 你会接收一段文本, 请将该文本生成摘要。"},
            {"role": "user", "content": text}
        ]
        prompt = tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
        batch_inputs.append(prompt)
        batch_targets.append(sample['title']+tokenizer.eos_token)

    # 编码：加了提示的原文本
    batch_data = tokenizer(
        batch_inputs,
        max_length=max_length,  # 统一使用相同的长度
        padding='max_length',
        truncation=True,
        return_tensors='pt').to(device)


    # 标签数据
    labels = tokenizer(
        batch_targets,
        max_length=max_length,
        padding='max_length',
        truncation=True,
        return_tensors="pt")['input_ids'].to(device)

    # 将 <eos> 后的 token 设置为 -100，避免计算损失
    # [1]:取列索引
    # eos_token_id:151645,pad_token_id:151643
    # 在计算交叉熵损失时，pad_token_id不用管，因为CrossEntropyLoss 通常会自动忽略填充标记的损失计算
    # 保险的措施：将 <pad> token 也设置为 -100，计算损失时一并忽略
    # labels[labels == tokenizer.pad_token_id] = -100
    end_token_index = torch.where(labels == tokenizer.eos_token_id )[1]
    for idx, end_idx in enumerate(end_token_index):
        labels[idx][end_idx+1:] = -100
    batch_data['labels'] = labels

    return batch_data

train_dataloader = DataLoader(train_data,batch_size=16,shuffle=True,collate_fn=collote_fn)
valid_dataloader = DataLoader(valid_data,batch_size=16,shuffle=False,collate_fn=collote_fn)
test_dataloader = DataLoader(test_data,batch_size=16,shuffle=False,collate_fn=collote_fn)

# 打印测试
batch = next(iter(train_dataloader))
print(batch.keys())
print('batch shape:', {k: v.shape for k, v in batch.items()})
# print(f"eos_token_id: {tokenizer.eos_token_id}")  # eos_token_id:151645
# print(f"pad_token_id:{tokenizer.pad_token_id}")  # 151643
print(batch)

