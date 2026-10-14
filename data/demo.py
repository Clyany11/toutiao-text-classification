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

print('id_to_label:', id_to_label)

