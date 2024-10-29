from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from transformers import AdamW, get_scheduler
import json,torch,os
import torch.nn as nn
import numpy as np
import random
# from Qwenfinetuning.model import model


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
# 使用梯度裁剪避免梯度爆炸：
# torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

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
            f.write('\n')
