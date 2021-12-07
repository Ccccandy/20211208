# -*- coding:utf-8 -*-
"""
构建概率图

# goal_set.p 已划分train和dev
# goal_set_simul.p 就是test
# 其中 train = 1882 ，dev = 268    
# 数据类型：{'consult_id': '10742613', 'disease_tag': '小儿支气管炎', 'goal': {'explicit_inform_slots': {'痰': '1'}, 'implicit_inform_slots': {'嗓子沙哑': '2'}}}

# 制作6大疾病和症状/检查之间的条件概率
"""
import numpy as np
import pickle
import sys
import os
# 手动chdir到指定目录
os.chdir(sys.path[0])

goal_set_all = pickle.load(file=open('./data/goal_set.p', "rb"))
goal_set_all_data = goal_set_all['train']+goal_set_all['dev']

# 制作slot_set.p
d_zqgy_set = dict()
d_zqgy_set_set = set()
d_fx_set = dict()
d_fx_set_set = set()
d_fr_set = dict()
d_fr_set_set = set()
d_gm_set = dict()
d_gm_set_set = set()
d_shxdgr_set = dict()
d_shxdgr_set_set = set()
d_xhbl_set = dict()
d_xhbl_set_set = set()

for i in goal_set_all_data:
    if i['disease_tag'] == '小儿支气管炎':
        for j in i['goal']['explicit_inform_slots'].keys():
            d_zqgy_set_set.add(j)
        for j in i['goal']['implicit_inform_slots'].keys():
            d_zqgy_set_set.add(j)
    elif i['disease_tag'] == '小儿腹泻':
        for j in i['goal']['explicit_inform_slots'].keys():
            d_fx_set_set.add(j)
        for j in i['goal']['implicit_inform_slots'].keys():
            d_fx_set_set.add(j)
    elif i['disease_tag'] == '小儿发热':
        for j in i['goal']['explicit_inform_slots'].keys():
            d_fr_set_set.add(j)
        for j in i['goal']['implicit_inform_slots'].keys():
            d_fr_set_set.add(j)
    elif i['disease_tag'] == '小儿感冒':
        for j in i['goal']['explicit_inform_slots'].keys():
            d_gm_set_set.add(j)
        for j in i['goal']['implicit_inform_slots'].keys():
            d_gm_set_set.add(j)
    elif i['disease_tag'] == '上呼吸道感染':  
        for j in i['goal']['explicit_inform_slots'].keys():
            d_shxdgr_set_set.add(j)
        for j in i['goal']['implicit_inform_slots'].keys():
            d_shxdgr_set_set.add(j)
    elif i['disease_tag'] == '小儿消化不良':
        for j in i['goal']['explicit_inform_slots'].keys():
            d_xhbl_set_set.add(j)
        for j in i['goal']['implicit_inform_slots'].keys():
            d_xhbl_set_set.add(j)
    

for index, value in enumerate(d_zqgy_set_set):
    d_zqgy_set[value] = index
for index, value in enumerate(d_fx_set_set):
    d_fx_set[value] = index
for index, value in enumerate(d_fr_set_set):
    d_fr_set[value] = index
for index, value in enumerate(d_gm_set_set):
    d_gm_set[value] = index
for index, value in enumerate(d_shxdgr_set_set):
    d_shxdgr_set[value] = index
for index, value in enumerate(d_xhbl_set_set):
    d_xhbl_set[value] = index


pickle.dump(d_zqgy_set, open('./data/d_zqgy_set.p', "wb"))
pickle.dump(d_fx_set, open('./data/d_fx_set.p', "wb"))
pickle.dump(d_fr_set, open('./data/d_fr_set.p', "wb"))
pickle.dump(d_gm_set, open('./data/d_gm_set.p', "wb"))
pickle.dump(d_shxdgr_set, open('./data/d_shxdgr_set.p', "wb"))
pickle.dump(d_xhbl_set, open('./data/d_xhbl_set.p', "wb"))





