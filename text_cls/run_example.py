# 将数据集之外的步骤也填补进来，作为参考，这里是一个完整版

# 1.构建数据集
# 1.1）加载数据集

from torch.utils.data import DataLoader,Dataset
import json
# 类别标签映射字典
label_mapping = {
    0: 'finance', 1: 'realty', 2: 'stocks', 3: 'education', 4: 'science',
    5: 'society', 6: 'politics', 7: 'sports', 8: 'game', 9: 'entertainment'
}

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

proprecess("E:\\NLP任务\\分类\\文本分类\\data\\THUCNews\\train.txt",
           "E:\\NLP任务\\分类\\文本分类\\data\\THUCNews\\processed_train.txt")
proprecess("E:\\NLP任务\\分类\\文本分类\\data\\THUCNews\\dev.txt",
           "E:\\NLP任务\\分类\\文本分类\\data\\THUCNews\\processed_dev.txt")
proprecess("E:\\NLP任务\\分类\\文本分类\\data\\THUCNews\\test.txt",
           "E:\\NLP任务\\分类\\文本分类\\data\\THUCNews\\processed_test.txt")


class THUCNews(Dataset):
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


train_data = THUCNews("E:\\NLP任务\\分类\\文本分类\\data\\THUCNews\\processed_train.txt")
valid_data = THUCNews("E:\\NLP任务\\分类\\文本分类\\data\\THUCNews\\processed_dev.txt")
test_data = THUCNews("E:\\NLP任务\\分类\\文本分类\\data\\THUCNews\\processed_test.txt")


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
max_length = 72
def collote_fn(batch_samples):
    batch_comment,batch_labels = [],[]
    for sample in batch_samples:
        comment = sample['comment']
        messages = [
            {"role": "system",
             "content": "你是一个文本分类的专家, 你会接收一段文本, 它的待选类别如下：\
            ['finance','realty','stocks','education','science','society','politics','sports','game','entertainment',],\
            请从以上的类别列表中选其一作为该文本的类别。"},
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



train_dataloader = DataLoader(train_data,batch_size=8,shuffle=True,collate_fn=collote_fn)
valid_dataloader = DataLoader(valid_data,batch_size=8,shuffle=False,collate_fn=collote_fn)
test_dataloader = DataLoader(test_data,batch_size=8,shuffle=False,collate_fn=collote_fn)

# 打印测试
batch = next(iter(train_dataloader))
print(batch.keys())
print('batch shape:', {k: v.shape for k, v in batch.items()})
print(batch)


# 3.模型训练与验证
# 3.1）训练函数
from tqdm.auto import tqdm
def train_loop(dataloader,model,optimizer,lr_scheduler,epoch, total_loss):
    total_loss = 0.
    total = 0
    model = model.train()
    progress_bar = tqdm(enumerate(dataloader), total=len(dataloader))
    for step, batch_data in progress_bar:
        model_inputs = batch_data['input_ids']
        attention_mask = batch_data['attention_mask']
        labels = batch_data['labels']
        outputs = model(
            input_ids=model_inputs,
            attention_mask=attention_mask,
            labels=labels)
        loss = outputs.loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        total_loss += loss.item()
        avg_loss = total_loss / (step + 1)
        progress_bar.set_description(f'loss:{avg_loss:>7f}')
    return total_loss

# 3.2) 测试函数
from rouge import Rouge
import random
import numpy as np

rouge = Rouge()
def test_loop(dataloader, model, mode='Valid'):
    assert mode in ['Valid', 'Test']
    model.eval()

    preds, labels = [], []
    for batch_data in dataloader:
        model_inputs = batch_data['input_ids']
        attention_mask = batch_data['attention_mask']
        labels = batch_data['labels']

        with torch.no_grad():
            generated_ids = model.generate(  # 1.生成预测
                model_inputs,
                attention_mask=attention_mask,
                max_new_tokens=512,
                num_beams=4,  # 使用柱搜索
                no_repeat_ngram_size=2, ).cpu().numpy()
        if isinstance(generated_ids, tuple):
            generated_ids = generated_ids[0]
        # 2.对预测解码
        decoded_preds = [
            output_ids[len(input_ids):] for input_ids,output_ids in zip(model_inputs,generated_ids)
        ]
        # 或者：
        # decoded_preds = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

        label_tokens = labels.cpu().numpy()
        label_tokens = np.where(labels != -100, label_tokens, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(label_tokens, skip_special_tokens=True)

        # 用空格连接结果用于匹配rouge格式
        preds += [' '.join(pred.strip()) for pred in decoded_preds]
        labels += [' '.join(label.strip()) for label in decoded_labels]

    scores = rouge.get_scores(hyps=preds, refs=labels, avg=True)
    result = {key: value['f'] * 100 for key, value in scores.items()}
    result['avg'] = np.mean(list(result.values()))
    print(f"{mode} Rouge1:{result['rouge-1']:>0.2f} Rouge2:{result['rouge-2']:>0.2f} \
            RougeL:{result['rouge-l']:>0.2f}\n")
    return result


# 3.3) 主循环
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from transformers import AdamW, get_scheduler
import json,os


def seed_everything(seed=1029):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
seed_everything(42)

learning_rate = 1e-4
epoch_num = 3

optimizer = AdamW(model.parameters(), lr=learning_rate)
lr_scheduler = get_scheduler(
    'linear',
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=epoch_num * len(train_dataloader))


best_avg_rouge = 0.
for epoch in range(epoch_num):
    print(f'Epoch {epoch + 1}/{epoch_num}\n------------------------------------')
    total_loss =0.  # 重置损失
    total_loss = train_loop(train_dataloader, model, optimizer, lr_scheduler, epoch + 1, total_loss)
    valid_rouge = test_loop(valid_dataloader, model, mode='Valid')
    rouge_avg = valid_rouge['avg']

    # 保存最佳模型权重
    if rouge_avg > best_avg_rouge:
        best_avg_rouge = rouge_avg
        print('saving new weights...\n')
        torch.save(model.state_dict(),
                   f'epoch_{epoch + 1}_valid_rouge_{rouge_avg:0.4f}_model_weights.bin')
        # 打印验证集评价指标
        print(f'rouge_avg:{rouge_avg}')

        # 将验证集指标记录到文件
        with open('rouge_avg.json', 'a') as f:
            json.dump({'epoch': epoch + 1, 'rouge': rouge_avg}, f)
            f.write('\n')  # 确保在文件关闭前执行写入操作,添加换行符便于文件读取

# 4.模型预测
model.load_state_dict(torch.load('***'))
model.eval()

with torch.no_grad():
    print('evaluating on test set...')
    sources, preds, labels = [], [], []
    for batch_data in test_dataloader:
        model_inputs = batch_data['input_ids']
        attention_mask = batch_data['attention_mask']
        labels = batch_data['labels']
        generated_ids = model.generate(  # 1.生成预测
            model_inputs,
            attention_mak=attention_mask,
            max_length=max_length,
            num_beams=4,
            no_repeat_ngram_size=2).cpu().numpy()
        if isinstance(generated_ids, tuple):
            generated_ids = generated_ids[0]
        # 2.对预测解码
        decoded_preds = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]

        # 转换标签并解码
        label_tokens = labels.cpu().numpy()
        label_tokens = np.where(labels != -100, label_tokens, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(label_tokens, skip_special_tokens=True)

        decoded_sources = tokenizer.batch_decode(
            model_inputs.cpu().numpy(),
            skip_special_tokens=True,
            use_source_tokenizer=True)

        preds += [' '.join(pred.strip()) for pred in decoded_preds]
        labels += [' '.join(label.strip()) for label in decoded_labels]
        sources += [' '.join(source.strip()) for source in decoded_sources]
    scores = rouge.get_scores(
        hyps=preds, refs=labels, avg=True)
    rouges = {key: value['f'] * 100 for key, value in scores.items()}
    rouges['avg'] = np.mean(list(rouges.values()))
    print(
        f"Test Rouge1: {rouges['rouge-1']:>0.2f} Rouge2: {rouges['rouge-2']:>0.2f} RougeL: {rouges['rouge-l']:>0.2f}\n")
    results = []
    for source, pred, label in zip(sources, preds, labels):
        results.append({
            'document': source,
            'prediction': pred,
            'summarization': label
        })
    with open('test_data_pred.json', 'wt', encoding='utf-8') as f:
        for example_result in results:
            f.write(json.dumps(example_result, ensure_ascii=False) + '\n')


