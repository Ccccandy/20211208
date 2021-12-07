import os
import json
import pickle
import numpy as np

slot_set = pickle.load(file=open('data2/slot_set.p',"rb"))
disease_set = pickle.load(file=open('data2/disease_set.p',"rb"))


# 症状识别-工具



def get_vocab():
    """获得字典"""
    w2i_char, i2w_char = load_vocabulary(os.path.join('data1', "vocab_char.txt"))  # 单词表
    w2i_bio, i2w_bio = load_vocabulary(os.path.join('data1', "vocab_bio.txt"))  # BIO表
    w2i_attr, i2w_attr = load_vocabulary(os.path.join('data1',"vocab_attr.txt"))  # 实体归一化 [咳嗽 咳嗽 null null null null]
    w2i_type, i2w_type = load_vocabulary(os.path.join('data1', "vocab_type.txt")) # 实体属性 [1 1 null null null null null]
    vocab_dict = {
        "w2i_char": w2i_char,
        "i2w_char": i2w_char,
        "w2i_bio": w2i_bio,
        "i2w_bio": i2w_bio,
        "w2i_attr": w2i_attr,
        "i2w_attr": i2w_attr,
        "w2i_type": w2i_type,
        "i2w_type": i2w_type
    }
    return vocab_dict


def load_vocabulary(path):
    """生成辅助字典"""
    vocab = open(path, "r", encoding="utf-8").read().strip().split("\n")
    print("load vocab from: {}, containing words: {}".format(path, len(vocab)))
    w2i = {}
    i2w = {}
    for i, w in enumerate(vocab):
        w2i[w] = i
        i2w[i] = w
    return w2i, i2w

def pair(bio_seq, word_seq, type_seq):
    assert len(bio_seq) == len(word_seq) == len(type_seq)
    with open('./data1/total_dic.json', 'r', encoding='utf-8') as fr:
        char2norm = json.load(fr)

    pair = set()
    pairs = set()
    v = ""
    for i in range(len(bio_seq)):
        word = word_seq[i]
        bio = bio_seq[i]
        type = type_seq[i]

        if bio == "O":
            v = ""
        elif bio == "S":
            v = word
            pair.add((type, v))
            v = ""
        elif bio == "B":
            v = word
        elif bio == "I":
            if v != "":
                v += word
        elif bio == "E":
            if v != "":
                v += word
                pair.add((type, v))
            v = ""
    for i in pair:
        type = i[0]
        v = i[1]
        attr = char2norm.get(v,'null')
        pairs.add((attr, type, v))

    return pairs



# 疾病诊断--工具

def state_to_rep(state,turn):

    current_slots_rep = np.zeros(len(slot_set.keys()))
    for slot in state.keys():
        if state[slot] == 1:
            if slot in slot_set.keys():
                current_slots_rep[slot_set[slot]] = 1.0              
        elif state[slot] == 2:
            if slot in slot_set.keys():
                current_slots_rep[slot_set[slot]] = -1.0             
        elif state[slot] == 3:
            if slot in slot_set.keys():
                current_slots_rep[slot_set[slot]] = -2.0
    turn_rep = np.zeros(22)
    turn_rep[turn] = 1.0
    # shape =(Len(slot_set)+max_turn,1)
    state_rep = np.hstack((current_slots_rep, turn_rep))
    return state_rep  
       
def build_action_space():

        action_spaces = []
        for slot in slot_set.keys():
            action_spaces.append(slot)
        for disease in disease_set.keys():
            action_spaces.append(disease)

        return action_spaces