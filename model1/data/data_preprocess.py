# -*- coding:utf-8 -*-
import pandas as pd
import numpy as np
import json
import os,sys
# 手动chdir到指定目录
os.chdir(sys.path[0])

def read_data(fn):
    """读取json数据"""
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

    train_num = int(all_num*0.9)

    train = shuffle_index[:train_num]
    dev = shuffle_index[train_num:]

    train_index = all_index[train]
    dev_index = all_index[dev]

    test_data = {}
    for i in dev_index:
        test_data[str(i)] = data[str(i)]

    with open('test.json', 'w', encoding='utf-8') as f:
        json.dump(test_data,f,ensure_ascii=False)
  
    return train_index,dev_index
 


def save_train_data(data, index,mode, fn1, fn2):
    """
    训练集和验证集的数据转换
    :param data: 用于训练的json数据
    :param example_ids: 样本id划分数据
    :param mode: train/dev
    :param fn1: 文本序列 input.seq.char
    :param fn2: BIO序列标签 output.seq.bio
    :return:
    ：每一行放入每一句的字符，且每个字符中有空格相连
    ：每一行放入每一句的BIO，且每个字符中有空格相连
    """
    seq_in, seq_bio = [], []
    for i in index:
        tmp_data = data[str(i)]
        tmp_dialogue = tmp_data['dialogue']
        for i in range(len(tmp_dialogue)):
            tmp_sent = list(tmp_dialogue[i]['speaker'] + '：' + tmp_dialogue[i]['sentence'])
            tmp_bio = ['O'] * 3 + tmp_dialogue[i]['BIO_label'].split(' ')
            if len(tmp_sent) != len(tmp_bio):
                print(tmp_sent)
            assert len(tmp_sent) == len(tmp_bio)
            seq_in.append(tmp_sent)
            seq_bio.append(tmp_bio)
    assert len(seq_in) == len(seq_bio)
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

def get_vocab_bio(fr1, fr2, fw):
    """获得bio种类字典"""
    bio = []
    with open(fr1, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            line = line.strip().split(' ')
            for i in line:
                if i not in bio:
                    bio.append(i)
    with open(fr2, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            line = line.strip().split(' ')
            for i in line:
                if i not in bio:
                    bio.append(i)

    bio = sorted(list(bio), key=lambda x: (x[2:], x[:2]))
    print('bio种类：', len(bio))

    with open(fw, 'w', encoding='utf-8') as f:
        for w in bio:
            f.write(w + '\n')



if __name__ == "__main__":

    # 获取所有数据
    all_data = read_data('./train.json')
    print(len(all_data))
    # 将数据进行分类，按照8：1：1划分
    train_index,dev_index = split_data(all_data)


    data_dir = 'ner_data'
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        os.makedirs(data_dir+'/train')
        os.makedirs(data_dir+'/dev')


    # 获得训练数据
    save_train_data(
        all_data,
        train_index,
        'train',
        os.path.join(data_dir, 'train', 'input.seq.char'),
        os.path.join(data_dir, 'train', 'output.seq.bio')
    )
    # 获得验证数据
    save_train_data(
        all_data,
        dev_index,
        'dev',
        os.path.join(data_dir, 'dev', 'input.seq.char'),
        os.path.join(data_dir, 'dev', 'output.seq.bio')
    )



    # 获取一些vocab信息
    get_vocab_bio(
        os.path.join(data_dir, 'train', 'output.seq.bio'),
        os.path.join(data_dir, 'dev', 'output.seq.bio'),
        os.path.join(data_dir, 'vocab_bio.txt')
    )

    get_vocab_char(
        os.path.join(data_dir, 'train', 'input.seq.char'),
        os.path.join(data_dir, 'dev', 'input.seq.char'),
        os.path.join(data_dir, 'vocab_char.txt')
    )
