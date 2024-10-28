import chardet

file_path = 'E:\\NLP任务\\生成式任务\\data\\生成式问答\\中文医疗问答数据\\总数据\\儿科5-14000.csv'

with open(file_path,'rb') as f:
    result = chardet.detect(f.read(10000))  # 读取前10,000字节进行检测
    print(result)  # 虽然检测结果是GB2312，但是最终确定gb18030，因为它可以读取已有的内科txt数据，其他的可能不行

# 对6个csv文件(即6个类别:内科，外科，妇产科，儿科，男科，肿瘤科)进行统一处理，并分割成训练集，验证集，测试集
# 1.总数据：2500条*6=15000
# 2.分割成训练集：12000，验证集：1500，测试集：1500，比例：8:1:1
# 3.每份数据集中都要包含6种疾病类别，比如儿科，内科等，即分布均匀
# 步骤：
# 1.读取每个 CSV 文件并抽取前 2500 条问答对,先不考虑department
# 2.对每个 CSV 文件的数据按照 8:1:1 的比例分割，并将分割结果分别加入到训练集、验证集和测试集列表中
# 3.最后，将训练集、验证集和测试集的数据写入各自的文件

import pandas as pd
import os
from sklearn.model_selection import train_test_split

# 设置数据路径和保存路径
data_path = "E:\\NLP任务\\生成式任务\\data\\生成式问答\\中文医疗问答数据\\总数据\\"
save_path = "E:\\NLP任务\\生成式任务\\data\\生成式问答\\中文医疗问答数据\\文本文件\\"
os.makedirs(save_path, exist_ok=True)

# 合并后保存的文件名
train_file = "train_data.txt"
valid_file = "dev_data.txt"
test_file = "test_data.txt"

# 获取所有csv文件
csv_files = [f for f in os.listdir(data_path) if f.endswith('.csv')]

# 用于存储训练，验证，测试集的问答对
train_qa_pairs, val_qa_pairs, test_qa_pairs = [], [], []

# 每个类别的数据读取和分割：
for csv_file in csv_files:
    try:
        # 读取csv文件并选择前2500条,尝试使用不同的编码
        df = pd.read_csv(os.path.join(data_path, csv_file), encoding='gb18030', errors='ignore')
    except Exception as e:  # except UnicodeDecodeError:
        print(f"Error reading {csv_file}: {e}")
        continue  # 跳过当前文件，继续下一个文件
        # 如果 UTF-8 解码失败，尝试使用 GBK 编码
        # df = pd.read_csv(os.path.join(data_path,csv_file),encoding='gbk')

    selected_rows = df.iloc[:2500]

    # 生成问答对格式的数据
    qa_pairs = [f"{{'ask':'{row['ask']}','answer':'{row['answer']}'}}\n"
                for _, row in selected_rows.iterrows()]

    # 按照8:1:1比例分割数据
    train_pairs, temp_pairs = train_test_split(qa_pairs, test_size=0.2, random_state=42)
    val_pairs, test_pairs = train_test_split(temp_pairs, test_size=0.5, random_state=42)

    # 添加到各自的列表中
    train_qa_pairs.extend(train_pairs)
    val_qa_pairs.extend(val_pairs)
    test_qa_pairs.extend(test_pairs)

# 将问答对写入各自的文件
# 1.训练集
with open(os.path.join(save_path, train_file), 'w', encoding='utf-8') as f:
    f.writelines(train_qa_pairs)

# 2.验证集
with open(os.path.join(save_path, valid_file), 'w', encoding='utf-8') as f:
    f.writelines(val_qa_pairs)

# 3.测试集
with open(os.path.join(save_path, test_file), 'w', encoding='utf-8') as f:
    f.writelines(test_qa_pairs)

print("数据分割完成，训练集、验证集和测试集已保存！")

