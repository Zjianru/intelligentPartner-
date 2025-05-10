# models/dialogue_model.py
from transformers import BertTokenizer, BertForSequenceClassification
import torch


class DialogueModel:
    def __init__(self):
        # 加载微调后的模型和分词器
        self.tokenizer = BertTokenizer.from_pretrained("./fine_tuned_model")
        self.model = BertForSequenceClassification.from_pretrained("./fine_tuned_model")

    def generate_response(self, text):
        # 将输入文本编码为模型所需的格式
        inputs = self.tokenizer(text, return_tensors="pt")

        # 使用模型进行推理
        with torch.no_grad():
            outputs = self.model(**inputs)

        # 获取预测结果
        prediction = torch.argmax(outputs.logits, dim=-1).item()
        return prediction


# 示例
if __name__ == "__main__":
    dialogue_model = DialogueModel()
    response = dialogue_model.generate_response("你好！")
    print("AI Response:", response)
