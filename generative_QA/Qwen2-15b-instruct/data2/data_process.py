# 数据集：中文医疗问答数据集

from torch.utils.data import DataLoader,Dataset,random_split

import pandas as pd
import json,os
import chardet  # 自动检测编码

# 1.读取儿科数据
# 检测文件编码
# with open("E:\\NLP任务\\生成式任务\\data\\generative_QA\\中文医疗问答数据\\总数据\\儿科5-14000.csv",'rb') as f:
#     result = chardet.detect(f.read())
# encoding = result['encoding']
#
# # 使用检测到的编码读取CSV
# df = pd.read_csv(
#     "E:\\NLP任务\\生成式任务\\data\\generative_QA\\中文医疗问答数据\\总数据\\儿科5-14000.csv",encoding=encoding)
# ask_list = df['ask'].values
# answer_list = df['answer'].values
# print(len(ask_list),len(answer_list))


# 取部分文件数据:内科
max_data_size = 4000  # 暂时不用设置了，因为数据集总数不多
train_data_size = 3238
temp_data_size = 810
valid_data_size = 405
test_data_size = 405
class MedicalQA(Dataset):
    def __init__(self, data_file):
        self.data = self.load_data(data_file)

    def load_data(self, data_file):
        Data = {}
        with open(data_file, 'rt', encoding='gb18030') as f:
            for idx, pair in enumerate(f.read().split('\n\n')):
                if not pair:
                    break

                lines = pair.strip().split('\n')
                if len(lines) == 2:
                    ask = lines[0].strip()  # 问题
                    answer = lines[1].strip()  # 答案

                    Data[idx] = {
                        'ask': ask,
                        'answer': answer,
                    }
        return Data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

data = MedicalQA("E:\\NLP任务\\生成式任务\\data\\生成式问答\\中文医疗问答数据\\总数据\\内科.txt")
print("Dataset length:", len(data))  # 打印数据集长度:4048
train_data, temp_data = random_split(data, [train_data_size, temp_data_size])
valid_data, test_data = random_split(temp_data, [valid_data_size, test_data_size])
print(train_data[0])
print(valid_data[0])
print(test_data[0])
print(len(train_data))  # 3238
print(len(valid_data))  # 405
print(len(test_data))  # 405
# 错误：编码问题，原文件中使用了不同的编码格式，比如utf-8,gbk,GB2312,gb18030,else,.etc
# 最后选定了gb18030

# device设置
import torch
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print(f"device num:{torch.cuda.device_count()}")
    print(f"device name:{torch.cuda.get_device_name()}")
else:
    device = torch.device("cpu")
    print("No GPU available,using the CPU instead.")



# 1.2）调用模型
from transformers import AutoTokenizer,AutoModelForCausalLM
model_name = "E:/NLP任务/离线模型/Qwen2-1.5b"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name,torch_dtype="auto")
model = model.to(device)

# 1.3)分批
max_length = 184
def collote_fn(batch_samples):
    batch_comment,batch_labels = [],[]
    for sample in batch_samples:
        comment = sample['ask']
        messages = [
            {"role": "system",
             "content": "你擅长在医疗领域答疑解惑，请你根据问题给出恰当的回答！"},
            {"role": "user", "content": comment}
        ]
        prompt = tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
        batch_comment.append(prompt)
        batch_labels.append(sample['answer']+tokenizer.eos_token)

    # 编码：加了提示的原文本
    batch_data = tokenizer(
        batch_comment,
        max_length=max_length,  # 统一使用相同的长度
        padding='max_length',  # 填充到相同的长度，确保损失计算的维度匹配
        truncation=True,
        return_tensors='pt').to(device)

    # 标签数据
    labels = tokenizer(
        batch_labels,
        max_length=max_length,
        padding='max_length',  # 填充到相同的长度，确保损失计算的维度匹配
        truncation=True,
        return_tensors="pt")['input_ids'].to(device)

    end_token_index = torch.where(labels == tokenizer.eos_token_id)[1]
    for idx, end_idx in enumerate(end_token_index):
        labels[idx][end_idx + 1:] = -100
    batch_data['labels'] = labels

    return batch_data



train_dataloader = DataLoader(train_data,batch_size=16,shuffle=True,collate_fn=collote_fn)
valid_dataloader = DataLoader(valid_data,batch_size=16,shuffle=False,collate_fn=collote_fn)
test_dataloader = DataLoader(test_data,batch_size=16,shuffle=False,collate_fn=collote_fn)

# 打印测试
batch = next(iter(train_dataloader))
print(batch.keys())
print('batch shape:', {k: v.shape for k, v in batch.items()})
print(batch)







