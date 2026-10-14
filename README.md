# 中文资讯文本智能分类系统（预训练模型微调与蒸馏部署）

基于 **今日头条（toutiao）中文资讯** 的 10 分类文本分类系统，覆盖从 **FastText 基线 → BERT 预训练微调 → 知识蒸馏压缩 → HTTP 服务部署** 的完整工程链路，用于演示“精度 / 速度 / 部署成本”之间的权衡方案。

## 项目特点

- **多方案对比**：同一数据集上给出 FastText、BERT、蒸馏 textCNN 三种方案的实现与评估。
- **预训练微调**：基于 `bert-base-chinese`，演示冻结权重与全量微调两种微调策略。
- **知识蒸馏**：以 BERT 为教师、textCNN 为学生，用 KL 散度软目标 + 交叉熵硬目标完成蒸馏。
- **可部署**：FastText 模型通过 Flask 暴露 HTTP 接口，附客户端调用示例。

## 数据集

- 来源：今日头条资讯文本（`文本\t标签id` 格式）。
- 类别（共 10 类，见 [data/class.txt](data/class.txt)）：

  `finance` / `realty` / `stocks` / `education` / `science` / `society` / `politics` / `sports` / `game` / `entertainment`

- 划分：`train.txt` / `dev.txt` / `test.txt`，另有面向 FastText 的预处理版本 `*_fast.txt`（按字）与 `*_fast1.txt`（按词）。

## 目录结构

```
投满分/
├── data/                       # 数据与预处理脚本
│   ├── class.txt               # 类别名单
│   ├── train.txt / dev.txt / test.txt
│   ├── train_fast*.txt         # FastText 格式数据(按字/按词)
│   ├── preprocess.py           # 按字切分 -> FastText 格式
│   └── preprocess1.py          # jieba 按词切分 -> FastText 格式
├── fast_text/                  # 方案一: FastText 基线 + 服务部署
│   ├── fast_text_baseline2.py  # 训练(带 autotune 超参搜索)
│   ├── app.py                  # Flask HTTP 服务
│   └── text.py                 # 客户端调用示例
├── toutiao/                    # 方案二: BERT 微调
│   └── src/
│       ├── models/bert.py      # BERT + 全连接分类头(冻结 weight)
│       ├── run.py              # 训练入口
│       ├── train_eval.py       # 训练 / 评估 / 测试
│       ├── utils.py            # 数据集与迭代器
│       └── predict.py          # 单条文本推理
└── bert_distil/                # 方案三: 知识蒸馏(BERT -> textCNN)
    └── src/
        ├── models/bert.py      # 教师模型
        ├── models/textCNN.py   # 学生模型
        ├── run.py              # 训练入口(trainbert / train_kd)
        ├── train_eval.py       # 含 KD 损失 train_kd
        ├── test.py / test_eval.py  # 测试入口
        └── utils.py            # 含文本级词表构建 build_dataset_CNN
```

## 环境依赖

```bash
python >= 3.8
torch
transformers
jieba
fasttext
flask
requests
scikit-learn
tqdm
numpy
```

## 快速开始

> 注意：各 `Config` 中的数据 / 模型路径为原始训练机器上的绝对路径（如 `/mnt/workspace/...`、`/home/ec2-user/...`），运行前请改为本地实际路径。预训练模型需自行准备 `bert-base-chinese`。

### 1. 数据预处理（FastText）

```bash
cd data
python preprocess.py      # 按字切分，生成 train_fast.txt
python preprocess1.py     # jieba 按词切分，生成 train_fast1.txt
```

### 2. FastText 基线训练与部署

```bash
cd fast_text
python fast_text_baseline2.py   # 训练并保存 toutiao_fasttext_<时间戳>.bin

# 启动服务（需先在 app.py 中把模型路径改为本地 .bin 文件）
python app.py

# 另开终端测试
python text.py                  # POST 到 /v1/main_server/，返回分类结果
```

### 3. BERT 微调

```bash
cd toutiao/src
python run.py --model bert      # 训练 + 测试
python predict.py               # 单条文本预测
```

### 4. 知识蒸馏（BERT → textCNN）

```bash
cd bert_distil/src
python run.py --task trainbert  # (可选) 先训练教师 BERT
python run.py --task train_kd   # 用教师软标签蒸馏训练 textCNN 学生

python test.py --task test_bert # 评测教师
python test.py --task test_kd   # 评测学生
```

## 关键实现说明

- **BERT 微调**（[toutiao/src/models/bert.py](toutiao/src/models/bert.py)）：`BertModel` + `Linear(768, num_classes)`，`pad_size=32`、`lr=5e-5`、`batch_size=128`。
- **蒸馏损失**（[bert_distil/src/train_eval.py](bert_distil/src/train_eval.py)）：`T=2`、`alpha=0.8`，
  `Loss = KLDiv(log_softmax(student/T), softmax(teacher/T)) * alpha * T² + CE(student, label) * (1 - alpha)`。
- **学生模型**（[bert_distil/src/models/textCNN.py](bert_distil/src/models/textCNN.py)）：字向量维度 300，卷积核 `(2,3,4) × 512`，dropout 0.5。
- **服务接口**（[fast_text/app.py](fast_text/app.py)）：`POST /v1/main_server/`，表单字段 `uid`、`text`，返回预测类别名。

## 说明

- 训练产物（`*.bin` / `*.pt` 及 `srcsaved_dict/`）体积较大（部分超过 GitHub 单文件 100MB 限制），已在 `.gitignore` 中排除，需要时请在本地重新训练或单独分发。
