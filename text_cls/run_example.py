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
