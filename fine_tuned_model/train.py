from transformers import BertForSequenceClassification, BertTokenizer, Trainer, TrainingArguments
from datasets import Dataset
import json


# 加载数据集
# 修改后的数据加载函数
def load_dataset(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    train_data = []
    for item in data['fine_tuned_model']:
        # 构造双向对话上下文
        context = f"User: {item['user']}\nAssistant: {item['ai']}"
        encoding = tokenizer(
            context,
            truncation=True,
            padding='max_length',
            max_length=128,
            return_tensors='pt'
        )
        # 根据实际业务逻辑设置标签（示例：0-中性/1-正向）
        label = 0 if '?' in item['ai'] else 1  # 示例逻辑
        train_data.append({
            'input_ids': encoding['input_ids'][0],
            'attention_mask': encoding['attention_mask'][0],
            'labels': label
        })
    return Dataset.from_dict({
        'input_ids': [x['input_ids'] for x in train_data],
        'attention_mask': [x['attention_mask'] for x in train_data],
        'labels': [x['labels'] for x in train_data]
    })


# 加载预训练模型和分词器

model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)
# 加载训练数据
# 分割训练集和验证集
full_dataset = load_dataset("./data/dialogues.json")
split_dataset = full_dataset.train_test_split(test_size=0.2, seed=42)
train_dataset = split_dataset['train']
eval_dataset = split_dataset['test']

# 配置训练参数
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_dir='./logs',
    learning_rate=2e-5,
    weight_decay=0.01,  # 新增权重衰减
    load_best_model_at_end=True,  # 新增最佳模型保存
    optim="adamw_torch_fused",
    gradient_accumulation_steps=2,
    gradient_checkpointing=True,

)

# 创建 Trainer 实例
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)
if __name__ == '__main__':
    # 微调模型
    trainer.train()
    # 保存微调后的模型
    model.save_pretrained("./fine_tuned_model")
    tokenizer.save_pretrained("./fine_tuned_model")