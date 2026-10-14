import os
import sys
import jieba

id_to_label = {}

idx = 0
with open('class.txt', 'r', encoding='utf-8') as f1:
    for line in f1.readlines():
        line = line.strip('\n').strip()
        id_to_label[idx] = line
        idx += 1

count = 0
train_data = []
with open('train.txt', 'r', encoding='utf-8') as f2:
    for line in f2.readlines():
        line = line.strip('\n').strip()
        sentence, label = line.split('\t')

        # 1: 首先处理标签部分
        label_id = int(label)
        label_name = id_to_label[label_id]
        new_label = '__label__' + label_name

        # 2: 然后处理文本部分, 区别于之前的按字划分, 此处按词划分文本
        sent_char = ' '.join(jieba.lcut(sentence))

        # 3: 将文本和标签组合成fasttext规定的格式
        new_sentence = new_label + '\t' + sent_char
        train_data.append(new_sentence)

        count += 1
        if count % 10000 == 0:
            print('count=', count)


with open('train_fast1.txt', 'w', encoding='utf-8') as f3:
    for data in train_data:
        f3.write(data + '\n')

print('FastText测试数据预处理完毕!')


