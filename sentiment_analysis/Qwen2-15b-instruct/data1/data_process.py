# 情感分析本质上是分类任务，同文本分类相同，这里在prompt的处理上稍有不同
# 数据集处理：
# 数据集：中文情感分析语料库 ChnSentiCorp

from torch.utils.data import DataLoader,Dataset
import json
# 类别标签映射字典
label_mapping = {
    0: 'negative', 1: 'positive'}

# 将标签转换为类别后保存
def proprecess(data_file,output_file):
    processed_data = []
    with open(data_file,'rt',encoding='utf-8') as f:
        for line in f:
            items = line.strip().split('\t')
            assert len(items) == 2
            comment = items[0]
            # 将数字标签转换为类别标签
            label = label_mapping[int(items[1])]
            processed_data.append({'comment':comment,'label':label})

    # 保存为新文件
    with open(output_file,'w',encoding='utf-8') as out_f:
        for item in processed_data:
            out_f.write(json.dumps(item,ensure_ascii=False)+'\n')

proprecess("E:\\NLP任务\\分类\情感分析\\data\\chnsenticorp\\train.txt",
           "E:\\NLP任务\\分类\情感分析\\data\\chnsenticorp\\processed_train.txt")
proprecess("E:\\NLP任务\\分类\情感分析\\data\\chnsenticorp\\dev.txt",
           "E:\\NLP任务\\分类\情感分析\\data\\chnsenticorp\\processed_dev.txt")
proprecess("E:\\NLP任务\\分类\情感分析\\data\\chnsenticorp\\test.txt",
           "E:\\NLP任务\\分类\情感分析\\data\\chnsenticorp\\processed_test.txt")

class ChnSentiCrop(Dataset):
    def __init__(self,data_file):
        self.data = self.load_data(data_file)

    def load_data(self, data_file):
        Data = {}
        with open(data_file, 'rt', encoding='utf-8') as f:
            for idx, line in enumerate(f):
                item = json.loads(line.strip())
                Data[idx] = item

        return Data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


train_data = ChnSentiCrop("E:\\NLP任务\\分类\情感分析\\data\\chnsenticorp\\processed_train.txt")
valid_data = ChnSentiCrop("E:\\NLP任务\\分类\情感分析\\data\\chnsenticorp\\processed_dev.txt")
test_data = ChnSentiCrop("E:\\NLP任务\\分类\情感分析\\data\\chnsenticorp\\processed_test.txt")


# 打印测试
print(f'train set size:{len(train_data)}')
print(f'valid set size:{len(valid_data)}')
print(f'test set size:{len(test_data)}')
print(next(iter(train_data)))
print(next(iter(valid_data)))
print(next(iter(test_data)))


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
max_length = 154
def collote_fn(batch_samples):
    batch_comment,batch_labels = [],[]
    for sample in batch_samples:
        comment = sample['comment']
        messages = [
            {"role": "system",
             "content": "你是一个情感分析的专家, 你会接收一段文本, 请判断它的情感极性，如果是正面，积极，优秀的评价，请回答:positive,"
                        "反之，请回答:negative"},
            {"role": "user", "content": comment}
        ]
        prompt = tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
        batch_comment.append(prompt)
        batch_labels.append(sample['label']+tokenizer.eos_token)

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



train_dataloader = DataLoader(train_data,batch_size=64,shuffle=True,collate_fn=collote_fn)
valid_dataloader = DataLoader(valid_data,batch_size=64,shuffle=False,collate_fn=collote_fn)
test_dataloader = DataLoader(test_data,batch_size=64,shuffle=False,collate_fn=collote_fn)

# 打印测试
batch = next(iter(train_dataloader))
print(batch.keys())
print('batch shape:', {k: v.shape for k, v in batch.items()})
print(batch)