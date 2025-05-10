# fine_tuned_model.py
from transformers import BertForSequenceClassification, BertTokenizer, Trainer, TrainingArguments
from datasets import Dataset
import json


# 加载数据集
def load_dataset(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 将数据转换为 Hugging Face Dataset 格式
    train_data = []
    for item in data['fine_tuned_model']:
        input_text = f"User: {item['user']} AI: {item['ai']}"
        encoding = tokenizer.encode(input_text, truncation=True, padding='max_length', max_length=128)
        train_data.append({'input_ids': encoding, 'labels': encoding})
    return Dataset.from_dict(
        {'input_ids': [x['input_ids'] for x in train_data], 'labels': [x['labels'] for x in train_data]})


# 加载预训练模型和分词器
tokenizer = BertTokenizer.from_pretrained("/Users/xaegon/code/intelligentPartner/fine_tuned_model")
model = BertForSequenceClassification.from_pretrained("/Users/xaegon/code/intelligentPartner/fine_tuned_model", num_labels=2)

# 加载训练数据
train_dataset = load_dataset("data/dialogues.json")

# 配置训练参数
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_dir='./logs',
)

# 创建 Trainer 实例
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
)
if __name__ == '__main__':
    # 微调模型
    trainer.train()
    # 保存微调后的模型
    model.save_pretrained("./fine_tuned_model")
