# 创建口语化和归一化的词典

import pandas as pd
import json
import os
import os,sys
# 手动chdir到指定目录
os.chdir(sys.path[0])

def read_train_data(fn):
    """读取用于训练的json数据"""
    with open(fn, 'r', encoding='utf-8') as fr:
        data = json.load(fr)
    return data


old2new = {'O': 'O', 'B-Symptom': 'B', 'B-Drug': 'O', 'B-Drug_Category': 'O', 'B-Medical_Examination': 'O',
               'B-Operation': 'O','I-Symptom': 'I', 'I-Drug': 'O', 'I-Drug_Category': 'O', 'I-Medical_Examination': 'O',
               'I-Operation': 'O'}

total_dic = {}
# B-Symptom 
# implicit_info
data = read_train_data('train.json')
for i in data.keys():
    for j in data[i]['dialogue']:
        bio = j['BIO_label'].split(' ') # str=>list
        bio = [old2new[i] for i in bio]
        sym = j['sentence']   # str
        oral = []
        norm = j['symptom_norm']    # norm的list
        ddd = {}
        for num in range(len(bio)):
            if bio[num] == 'B':
                if num == len(bio)-1 or bio[num+1] == 'O' or bio[num+1] == 'B':
                    syms = sym[num]
                    oral.append(syms)
                    continue
                for rest in range(num+1,len(bio)):
                    if bio[rest] == 'I' and rest == len(bio)-1:
                        syms = sym[num:]
                        oral.append(syms)
                        break
                    if bio[rest] != 'I' :
                        syms = sym[num:rest]
                        oral.append(syms)
                        break
        ddd = dict(zip(oral,norm))
        total_dic=dict(total_dic, **ddd)

print(total_dic)
with open('total_dic.json', 'w', encoding='utf-8') as json_file:
    json.dump(total_dic, json_file, ensure_ascii=False, indent=4) 


