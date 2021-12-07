# -*- coding:utf-8 -*-

import numpy as np
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


def split_data(data):
    """划分数据集,获取id值"""
    train_index,dev_index = [],[]
    all_index = [i for i in data.keys()]
    all_index = np.array(all_index)
    all_num = len(all_index)
    shuffle_index = np.arange(all_num)
    #np.random.shuffle(shuffle_index)

    train_num = int(all_num*0.1)

    dev = shuffle_index[:train_num]
    train = shuffle_index[train_num:]

    train_index = all_index[train]
    dev_index = all_index[dev]

  
    return train_index,dev_index


def save_train_data(data, index, mode, fn1, fn2, fn3, fn4):
    """
    训练集和验证集的数据转换 (trick：本次处理时将BIO转化为BIOES标注，以提高多任务训练的整体准确率)
    :param data: 用于训练的json数据
    :param example_ids: 样本id划分数据
    :param mode: train/dev
    :param fn1: 文本序列 input.seq.char
    :param fn2: BIO序列标签 output.seq.bio
    :param fn3: 症状规范化标签 output.seq.attr
    :param fn4: 症状类别标签 output.seq.type
    :return:
    """
    # 只需要识别症状，因此将"B-Symptom"变为"B", 将"I-Symptom"变为"I",其余均为"O"
    old2new = {'O': 'O', 'B-Symptom': 'B', 'B-Drug': 'O', 'B-Drug_Category': 'O', 'B-Medical_Examination': 'O',
               'B-Operation': 'O','I-Symptom': 'I', 'I-Drug': 'O', 'I-Drug_Category': 'O', 'I-Medical_Examination': 'O',
               'I-Operation': 'O'}

    seq_in, seq_bio, seq_attr, seq_type = [], [], [], []
    for i in index:
        tmp_data = data[str(i)]
        tmp_dialogue = tmp_data['dialogue']
        for i in range(len(tmp_dialogue)):
            tmp_sent = list(tmp_dialogue[i]['speaker'] + '：' + tmp_dialogue[i]['sentence'])
            tmp_bio_origin = ['O'] * 3 + tmp_dialogue[i]['BIO_label'].split(' ')
            #print(tmp_sent)
            tmp_bio = [old2new[i] for i in tmp_bio_origin]
            assert len(tmp_sent) == len(tmp_bio)
            tmp_attrs = tmp_dialogue[i]['symptom_norm']
            tmp_types = tmp_dialogue[i]['symptom_type']
            # attr 症状规范化 + type 症状识别
            k = 0
            tmp_attr = ['null'] * len(tmp_bio)
            tmp_type = ['null'] * len(tmp_bio)
            j = 0
            while j < len(tmp_bio):
                if tmp_bio[j] == 'B':
                    k += 1
                    #print(j,k)
                    #print(i)
                    #print(tmp_sent)
                    #print(tmp_attr)
                    #print(tmp_attrs)
                    if tmp_types[0]=="21":
                        print(tmp_sent)
                    tmp_attr[j] = tmp_attrs[k - 1]
                    tmp_type[j] = tmp_types[k - 1]
                    m = j + 1
                    while m < len(tmp_bio):
                        if tmp_bio[m] == 'I':
                            tmp_attr[m] = tmp_attrs[k - 1]
                            tmp_type[m] = tmp_types[k - 1]
                            m += 1
                        else:
                            break
                    j = m
                else:
                    j += 1

            # BIO转化为BIOES
            label = tmp_bio
            j = 0
            while j + 1 < len(tmp_bio):
                if label[j] == 'B' and label[j + 1] == 'O':
                    label[j] = 'S'
                if label[j] == 'I' and label[j + 1] == 'O':
                    label[j] = 'E'
                if label[j] == 'I' and label[j + 1] == 'B':
                    label[j] = 'E'
                if label[j] == 'B' and label[j + 1] == 'B':
                    label[j] = 'S'
                j += 1
            if label[-1] == 'B':
                label[-1] = 'S'
            elif label[-1] == 'I':
                label[-1] = 'E'

            seq_in.append(tmp_sent)
            seq_bio.append(label)
            seq_attr.append(tmp_attr)
            seq_type.append(tmp_type)
    assert len(seq_in) == len(seq_bio) == len(seq_attr) == len(seq_type)
    print(mode, '句子数量为：', len(seq_in))
    # 数据保存
    with open(fn1, 'w', encoding='utf-8') as f1:
        for i in seq_in:
            tmp = ' '.join(i)
            f1.write(tmp+'\n')
    with open(fn2, 'w', encoding='utf-8') as f2:
        for i in seq_bio:
            tmp = ' '.join(i)
            f2.write(tmp+'\n')
    with open(fn3, 'w', encoding='utf-8') as f3:
        for i in seq_attr:
            tmp = ' '.join(i)
            f3.write(tmp+'\n')
    with open(fn4, 'w', encoding='utf-8') as f4:
        for i in seq_type:
            #print(i)
            tmp = ' '.join(i)
            f4.write(tmp+'\n')

def get_vocab_attr(fr, fw):
    """获得规范化后症状"""
    vocab_attr = pd.read_csv(fr)
    vocab_attr = vocab_attr['norm'].to_list()
    # 添加null
    add_tokens = ['null']
    vocab_attr = add_tokens + vocab_attr
    print('规范化后症状种类：', len(vocab_attr))
    with open(fw, 'w', encoding='utf-8') as f:
        for w in vocab_attr:
            f.write(w+ '\n')

def get_vocab_bio(fn):
    """获得BIOES标签字典"""
    tokens = ['O', 'I', 'B', 'E', 'S']
    print('BIOES标签类别：', len(tokens))
    with open(fn, 'w', encoding='utf-8') as f:
        for w in tokens:
            f.write(w + '\n')

def get_vocab_type(fn):
    """获得症状类别标签字典"""
    tokens = ['null', '0', '1', '2']
    print('症状类别：', len(tokens))
    with open(fn, 'w', encoding='utf-8') as f:
        for w in tokens:
            f.write(w + '\n')

def get_vocab_char(fr1, fr2, fw):
    """获得字符种类字典"""
    chars = []
    with open(fr1, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            line = line.strip().split(' ')
            for i in line:
                if i not in chars:
                    chars.append(i)
    with open(fr2, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            line = line.strip().split(' ')
            for i in line:
                if i not in chars:
                    chars.append(i)
    add_tokens = ['[PAD]', '[UNK]', '[SEP]', '[SPA]']
    chars = add_tokens + chars
    print('字符种类：', len(chars))

    with open(fw, 'w', encoding='utf-8') as f:
        for w in chars:
            f.write(w + '\n')


if __name__ == "__main__":

    all_data = read_train_data('train.json')

    print(len(all_data))
    # 将数据进行分类，按照9：1划分
    train_index,dev_index = split_data(all_data)


    
    data_dir = 'near_data'
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        os.makedirs(data_dir+'/train')
        os.makedirs(data_dir+'/dev')

    save_train_data(
        all_data,
        train_index,
        'train',
        os.path.join(data_dir, 'train', 'input.seq.char'),
        os.path.join(data_dir, 'train', 'output.seq.bio'),
        os.path.join(data_dir, 'train', 'output.seq.attr'),
        os.path.join(data_dir, 'train', 'output.seq.type')
    )
    save_train_data(
        all_data,
        dev_index,
        'dev',
        os.path.join(data_dir, 'dev', 'input.seq.char'),
        os.path.join(data_dir, 'dev', 'output.seq.bio'),
        os.path.join(data_dir, 'dev', 'output.seq.attr'),
        os.path.join(data_dir, 'dev', 'output.seq.type')
    )

    # 获取一些vocab信息
    get_vocab_attr(
        'symptom_norm.csv',
        os.path.join(data_dir, 'vocab_attr.txt')
    )

    get_vocab_bio(os.path.join(data_dir, 'vocab_bio.txt'))

    get_vocab_type(os.path.join(data_dir, 'vocab_type.txt'))

    get_vocab_char(
        os.path.join(data_dir, 'train', 'input.seq.char'),
        os.path.join(data_dir, 'dev', 'input.seq.char'),
        os.path.join(data_dir, 'vocab_char.txt')
    )
