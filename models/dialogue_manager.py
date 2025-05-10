# models/dialogue_manager.py
from models.dialogue_model import DialogueModel
from models.sentiment_model import SentimentAnalyzer


class DialogueManager:
    def __init__(self):
        self.history = ""  # 保存对话历史
        self.dialogue_generator = DialogueModel()
        self.sentiment_analyzer = SentimentAnalyzer()

    def handle_input(self, user_input):
        # 情感分析
        sentiment = self.sentiment_analyzer.analyze_sentiment(user_input)
        print("Detected Sentiment:", sentiment)

        # 将用户输入添加到对话历史中
        self.history += f"User: {user_input}\nAI:"

        # 使用对话生成模型生成回应
        response = self.dialogue_generator.generate_response(self.history)

        # 记录AI回应
        self.history += f" {response}\n"
        return str(response)


# 示例
if __name__ == "__main__":
    dialogue_manager = DialogueManager()
    user_input = "你好！今天怎么样？"
    print("AI Response:", dialogue_manager.handle_input(user_input))
    user_input = "我今天有点累"
    print("AI Response:", dialogue_manager.handle_input(user_input))
