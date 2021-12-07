import pickle
import json
import sys
import os
# 手动chdir到指定目录
os.chdir(sys.path[0])

import pandas as pd

norm = pd.read_csv('./data/symptom_norm.csv')
norm = norm['norm'].to_list()
print('归一化数量：',len(norm))

slot_set_all = pickle.load(file=open('./data/slot_set.p', "rb"))
slot = [i for i in slot_set_all.keys()]
print('症状/检查数量：',len(slot))

sym = [i for i in slot if i in norm]
print('症状数量：',len(sym))

ct = [i for i in slot if i not in norm]
print('检查数量：',len(ct))

# 保存检查
with open('./datact.json', 'w', encoding='utf-8') as json_file:
    json.dump(ct, json_file, ensure_ascii=False, indent=4) 
# 保存症状
with open('./datasym.json', 'w', encoding='utf-8') as json_file:
    json.dump(sym, json_file, ensure_ascii=False, indent=4) 

