# 智能对话伙伴系统

基于微调BERT模型的对话管理系统，支持上下文感知和个性化响应生成。

## 功能特性
- 上下文感知对话管理
- 用户意图分类（支持问询/陈述两种类型）
- 多轮对话支持
- 响应相似度匹配

## 安装指南
```bash
# 克隆项目
git clone https://github.com/yourrepo/intelligentPartner.git

# 安装依赖
pip install -r requirements.txt

# 准备数据
mkdir data && cp sample_dialogues.json data/dialogues.json
```

## 使用说明
### 模型训练
```bash
python fine_tuned_model/train.py
```

### 启动对话系统
```bash
python main.py
```

## 项目结构
```
intelligentPartner/
├── fine_tuned_model/    # 模型训练模块
│   ├── train.py         # 训练脚本
│   └── *.bin           # 预训练模型文件
├── models/              # 模型实现
│   ├── dialogue_model.py # 对话管理模型
│   └── sentiment_model.py # 情感分析模型
├── data/                # 数据目录
│   └── dialogues.json   # 对话数据集
├── main.py              # 主入口程序
└── requirements.txt     # 依赖清单
```
