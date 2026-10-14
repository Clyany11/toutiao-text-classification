import torch
import torch.nn as nn
import os
from transformers import BertModel, BertTokenizer, BertConfig


class Config(object):
    def __init__(self, dataset):
        self.model_name = "bert"
        self.data_path = "/mnt/workspace/AI/11投满分/data/"
        self.train_path = self.data_path + "train.txt"  # 训练集
        self.dev_path = self.data_path + "dev.txt"      # 验证集
        self.test_path = self.data_path + "test.txt"    # 测试集
        self.class_list = [x.strip() for x in open(self.data_path + "class.txt").readlines()]  # 类别名单
        self.save_path = '/mnt/workspace/AI/11投满分/toutiao/srcsaved_dict'
        if not os.path.exists(self.save_path):
            os.mkdir(self.save_path)
        self.save_path += "/" + self.model_name + ".pt"  # 模型训练结果

        # ========= 建议补充BERT常用配置参数（文本分类任务）=========
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.pad_size = 32        # 句子最大长度
        self.batch_size = 128
        self.learning_rate = 5e-5
        self.bert_path = '/mnt/workspace/AI/models/bert-base-chinese'  # 预训练bert模型名称/本地路径
        self.hidden_size = 768
        self.num_classes = len(self.class_list)
        self.num_epochs = 3
        self.require_improvement = 1000  # 若超过1000batch效果没提升，提前结束
        self.pad_token_id = 0
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_path)
        self.bert_config = BertConfig.from_pretrained(self.bert_path + '/config.json')

class Model(nn.Module):    
    def __init__(self, config):        
        super(Model, self).__init__()        
        self.bert = BertModel.from_pretrained(config.bert_path, config=config.bert_config) 
        for name, param in self.bert.named_parameters():
            if name.endswith('weight'):
                print(name)
                param.requires_grad = False  
        self.fc = nn.Linear(config.hidden_size, config.num_classes)    
    def forward(self, x):        
        # 输⼊的句⼦        
        context = x[0]              #文本
        # 对padding部分进⾏mask, 和句⼦⼀个size, padding部分⽤0表示, ⽐如[1, 1, 1, 1, 0, 0]        
        mask = x[2]                 #mask
        out = self.bert(context, attention_mask=mask)        
        out = self.fc(out.pooler_output)        
        return out

