# -*- coding:utf-8 -*-
"""
数据预处理

# goal_set.p 已划分train和dev
# goal_set_simul.p 就是test
# 其中 train = 1882 ，dev = 268
# 数据类型：{'consult_id': '10742613', 'disease_tag': '小儿支气管炎', 'goal': {'explicit_inform_slots': {'痰': '1'}, 'implicit_inform_slots': {'嗓子沙哑': '2'}}}

# 制作disease_set.p
# 制作slot_set.p
# 制作disease_symptom_set.p
"""

import pickle
import sys
import os
# 手动chdir到指定目录
os.chdir(sys.path[0])

# 制作disease_set.p
goal_set_all = pickle.load(file=open('./data/goal_set.p', "rb"))
goal_set_all_data = goal_set_all['train']+goal_set_all['dev']


#test = pickle.load(file=open('./data/goal_set_simul.p', "rb"))
#print('test:',len(test['test']))
#print('train:',goal_set_all['train'][0])
#print('dev:',len(goal_set_all['dev']))

a = []
for i in goal_set_all['train']:
    if i['disease_tag']=='小儿消化不良':
        a.append(0)
    elif i['disease_tag']=='小儿发热':
        a.append(1)
    elif i['disease_tag']=='小儿支气管炎':
        a.append(2)
    elif i['disease_tag']=='小儿感冒':
        a.append(3)
    elif i['disease_tag']=='上呼吸道感染':
        a.append(4)
    elif i['disease_tag']=='小儿腹泻':
        a.append(5)

print('小儿消化不良',a.count(0))
print('小儿发热',a.count(1))
print('小儿支气管炎',a.count(2))
print('小儿感冒',a.count(3))
print('上呼吸道感染',a.count(4))
print('小儿腹泻',a.count(5))

disease_set = dict()
disease_set_set = []
n = 0
for i in goal_set_all_data:
    if i['disease_tag'] not in disease_set_set:
        disease_set_set.append(i['disease_tag'])

for index, value in enumerate(disease_set_set):
    disease_set[value] = index

print(disease_set)
print(disease_set_set)
pickle.dump(disease_set, open('./data/disease_set.p', "wb"))

# 制作slot_set.p
slot_set = dict()
slot_set_set = []

for i in goal_set_all_data:
    for j in i['goal']['explicit_inform_slots'].keys():
        if j not in slot_set_set:
            slot_set_set.append(j)
    for j in i['goal']['implicit_inform_slots'].keys():
        if j not in slot_set_set:
            slot_set_set.append(j)

for index, value in enumerate(slot_set_set):
    slot_set[value] = index

print('slot',len(slot_set))
pickle.dump(slot_set, open('./data/slot_set.p', "wb"))

# 制作disease_symptom_set.p
disease_symptom_set = dict()
disease_set_list = list(disease_set_set)

# 构建sub_list
def Create_list(key):
    slot_list = []
    for i in goal_set_all_data:
        if i['disease_tag'] == key:
            for j in i['goal']['explicit_inform_slots'].keys():
                if i['goal']['explicit_inform_slots'][j]=='1':
                    slot_list.append(j)
            for j in i['goal']['implicit_inform_slots'].keys():
                if i['goal']['implicit_inform_slots'][j]=='1':
                    slot_list.append(j)
    return slot_list
# 构建sub_dict
def Count(List):
    dict={}
    for key in List:
        if key in dict.keys():
            dict[key]=dict[key]+1
        else:
            dict[key]=1
    return dict

for index,i in enumerate(disease_set_list):
    sub_list = Create_list(i)
    sub_dic = Count(sub_list)
    disease_symptom_set[i] = dict()
    disease_symptom_set[i]['index'] = index
    disease_symptom_set[i]['Symptom'] = sub_dic

#print(disease_symptom_set)
for key,value in disease_symptom_set.items():
    print(key)
    a = value['Symptom']
    print(len(a))
print(disease_symptom_set)
pickle.dump(disease_symptom_set, open('./data/disease_symptom_set.p', "wb"))





