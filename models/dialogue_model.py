# models/dialogue_model.py
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import json


class DialogueModel:
    def __init__(self):
        # 加载微调后的模型和分词器
        self.tokenizer = BertTokenizer.from_pretrained("./fine_tuned_model")
        self.model = BertForSequenceClassification.from_pretrained("./fine_tuned_model", output_hidden_states=True)

    def generate_response(self, text):
        # 将输入文本编码为模型所需的格式
        inputs = self.tokenizer(text, return_tensors="pt")

        # 使用模型进行推理
        with torch.no_grad():
            outputs = self.model(**inputs)

        # 获取预测结果
        # 标签到自然语言回复的映射
        # 加载对话数据集
        with open('./data/dialogues.json', 'r') as f:
            dialogues = json.load(f)['fine_tuned_model']
        
        # 计算输入与样本的相似度
        input_encoding = self.tokenizer(text, return_tensors='pt')
        best_score = -1
        best_response = "暂时无法理解您的请求"
        
        for sample in dialogues:
            sample_encoding = self.tokenizer(sample['user'], return_tensors='pt')
            similarity = torch.cosine_similarity(
                outputs.hidden_states[-1].mean(dim=1),
                self.model(**sample_encoding).hidden_states[-1].mean(dim=1)
            ).item()
            
            if similarity > best_score:
                best_score = similarity
                best_response = sample['ai']
        
        return best_response


# 示例
if __name__ == "__main__":
    dialogue_model = DialogueModel()
    response = dialogue_model.generate_response("你好！")
    print("AI Response:", response)
