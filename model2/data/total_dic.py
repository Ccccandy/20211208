import json
import os,sys
# 手动chdir到指定目录
os.chdir(sys.path[0])


with open('total_dic.json', 'r', encoding='utf-8') as fr:
    data = json.load(fr)

a = []
for i,j in data.items():
    a.append(j)
s = set(a)
s = list(s)
print(s)
print(len(s))
import pandas as pd

vocab_attr = pd.read_csv('symptom_norm.csv')
vocab_attr = vocab_attr['norm'].to_list()
print(len(vocab_attr))

jiaoji=[i for i in s if i in vocab_attr]
print(len(jiaoji))