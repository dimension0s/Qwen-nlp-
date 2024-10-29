from tqdm.auto import tqdm
from torch.cuda.amp import autocast, GradScaler
# 混合精度训练（提高计算效率和内存利用率）
scaler = GradScaler()

def train_loop(dataloader, model, optimizer, lr_scheduler, epoch, total_loss):
    model.train()
    total = 0

    progress_bar = tqdm(enumerate(dataloader), total=len(dataloader))
    for step, batch_data in progress_bar:
        # 检查Nan或Inf值：
        # if torch.isnan(batch_data['input_ids']).any() or torch.isinf(batch_data['input_ids']).any():
            # print("Data contains NaN or Inf values!")
            # continue  # 跳过当前 batch 或采取其他措施
        model_inputs = batch_data['input_ids']
        attention_mask = batch_data['attention_mask']
        labels = batch_data['labels']
        with autocast():         
            outputs = model(
                input_ids=model_inputs,
                attention_mask=attention_mask,
                labels=labels)
            loss = outputs.loss
        

        optimizer.zero_grad()
        # 1.做混合精度训练：
        scaler.scale(loss).backward()  # 缩放损失以避免 FP16 下的数值溢出
        scaler.step(optimizer)  # 更新权重
        scaler.update()  # 更新缩放因子
        # 2.不做混合精度训练：
        # loss.backward()
        # optimizer.step()
        lr_scheduler.step()

        total_loss += loss.item()
        avg_loss = total_loss / (step + 1)
        progress_bar.set_description(f'loss:{avg_loss:>7f}')
    return total_loss
