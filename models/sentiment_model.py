# models/sentiment_model.py
from transformers import BertForSequenceClassification, BertTokenizer
import torch


class SentimentAnalyzer:
    def __init__(self):
        # 加载情感分析模型和分词器
        self.tokenizer = BertTokenizer.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")
        self.model = BertForSequenceClassification.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")

    def analyze_sentiment(self, text):
        # 编码输入文本
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)

        # 使用模型进行情感分析
        outputs = self.model(**inputs)
        sentiment = torch.argmax(outputs.logits, dim=-1).item()

        return sentiment


# 示例
if __name__ == "__main__":
    sentiment_analyzer = SentimentAnalyzer()
    sentiment = sentiment_analyzer.analyze_sentiment("我今天心情不好")
    print("Sentiment:", sentiment)  # 输出情感标签（0: 负面，1: 中性，2: 正面等）
