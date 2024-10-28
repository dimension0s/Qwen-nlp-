from rouge import Rouge
import random,torch
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