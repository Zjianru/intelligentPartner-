# app.py
from models.dialogue_manager import DialogueManager


def main():
    print("AI 伴侣系统启动！")
    dialogue_manager = DialogueManager()

    while True:
        user_input = input("你: ")
        if user_input.lower() == "exit":
            print("退出对话")
            break
        response = dialogue_manager.handle_input(user_input)
        print("AI: " + response)


if __name__ == "__main__":
    main()
