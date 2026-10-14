import torch
from models.bert import Config, Model

# 1. 加载模型
config = Config('toutiao')
model = Model(config).to(config.device)
model.load_state_dict(torch.load(config.save_path, map_location=config.device))
model.eval()

# 2. 预测一条文本
text = "中国男足今晚迎来世界杯预选赛关键一战"

# 分词 + 转编号（和训练时一样的处理）
token = config.tokenizer.tokenize(text)
token = ['[CLS]'] + token
token_ids = config.tokenizer.convert_tokens_to_ids(token)

# padding 到 pad_size=32
if len(token_ids) < config.pad_size:
    mask = [1] * len(token_ids) + [0] * (config.pad_size - len(token_ids))
    token_ids += [0] * (config.pad_size - len(token_ids))
else:
    mask = [1] * config.pad_size
    token_ids = token_ids[:config.pad_size]

# 转 tensor，加 batch 维度
input_ids = torch.LongTensor([token_ids]).to(config.device)
attention_mask = torch.LongTensor([mask]).to(config.device)

# 预测
with torch.no_grad():
    outputs = model((input_ids, None, attention_mask))
    pred = torch.max(outputs.data, 1)[1].item()

print(f"预测类别编号: {pred}")
print(f"预测类别名称: {config.class_list[pred]}")