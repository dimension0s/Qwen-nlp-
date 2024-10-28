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