# 备份内容：

from transformers import AutoModelForCausalLM,AutoTokenizer
from transformers import Trainer,DataCollatorForSeq2Seq,TrainingArguments

args = TrainingArguments(
    per_device_train_batch_size=32,  # 要与 DataLoader 中的 batch_size 一致
    # 每4个批次才进行一次参数更新:允许你使用较小的批次大小，同时模拟更大批次的效果,有助于在内存受限的情况下训练大型模型
    gradient_accumulation_steps=4,
    logging_steps=10,  # 每10个步骤记录一次训练日志，控制了训练过程中日志输出的频率
    num_train_epochs=2,  # 设置训练的轮数（epoch）为2。整个训练数据集将被遍历两次
    save_steps=100,  # 每100个步骤保存一次模型检查点。这决定了模型保存的频率
    learning_rate=1e-4,
    save_total_limit=1,  # 仅保存最近的模型,有助于节省磁盘空间，只保留最新的模型
    save_on_each_node=True,  # 在分布式训练中，每个节点都会保存模型,如果不是分布式，该参数可以忽略
    output_dir="/Qwenfinetuning\\text_cls\\Qwen2-15b-instruct\\data1\\output",  # 模型保存的目录
    report_to='none',  # 不使用任何报告工具（如 TensorBoard、Weights & Biases 等）来记录训练过程
    # 如果你想使用这些工具，可以改为 'tensorboard' 或其他支持的选项
)
# 模型训练
trainer = Trainer(
    model=model,
    train_dataset=train_data,  # 使用 Dataset 对象，而不是 DataLoader
    eval_dataset=valid_data,   # 直接使用 valid_data 而非 DataLoader
    data_collator=DataCollatorForSeq2Seq(tokenizer,padding=True),
    args=args,
)
trainer.train()

# 预测函数
def predict(messages,model,tokenizer):
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    model_inputs = tokenizer([text],return_tensors="pt")
    generated_ids = model.generate(
        model_inputs.input_ids,
        max_new_tokens=512,
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids,skip_special_tokens=True)[0]
    print(response)
    return response

# 用测试集测试
for i in range(len(test_data)):
    comment = test_data[i]['comment']
    label = test_data[i]['label']

    prompt = "你是一个文本分类的专家, 你会接收一段文本, 它的待选类别如下：\
            ['finance','realty','stocks','education','science','society','politics','sports','game','entertainment',],\
            请从以上的类别列表中选其一作为该文本的类别。"
    messages = [
        {'role':'system','content':f"{prompt}"},
        {'role':'user','content':f"{comment}"}
    ]

    response = predict(messages,model,tokenizer)
    messages.append({"role": "assistant", "content": f"{response}"})
    result_text = f"{messages[0]}\n\n{messages[1]}\n\n{messages[2]}"
    print(result_text)