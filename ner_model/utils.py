import codecs
import math
import os
import time
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import pandas as pd
import numpy as np

from bert import tokenization


def read_data(input_file):
    """Reads a BIO data.
    Args:
        input_file: 输入的文件名
    Outputs:
        lines: 样本的文本序列与BIO标注序列构成的列表的列表
           e.g. [{text:['机','械','设','计','基','础','的','作','者','是','谁','？'],label:['B','I','I','I','I','I','O','O','O','O','O','O','O','O']}, {...}, {...}]
    """
    with codecs.open(input_file, 'r', encoding='utf-8') as f:
        lines = []
        words = []
        labels = []
        for line in f:
            contends = line.strip()
            tokens = contends.split(' ')
            if len(tokens) == 2:  # 如果读入的一行有2列 token | label
                words.append(tokens[0])
                labels.append(tokens[-1])
            else:  # 如果读入的是空行，就认为已读入一个句子，将其加入到lines数组
                if len(contends) == 0 and len(words) > 0:  # 如果读入的是空行,且之前读入过行
                    lines.append({'text':words, 'label':labels})
                    words = []
                    labels = []
                    continue
    return lines

def timeSince(since):
    # 功能:获取每次打印的时间消耗, since是训练开始的时间
    # 获取当前的时间
    now = time.time()

    # 获取时间差, 就是时间消耗
    s = now - since

    # 获取时间差的分钟数
    m = math.floor(s / 60)

    # 获取时间差的秒数
    s -= m * 60

    return '%dm %ds' % (m, s)


def accuracy(out, labels):
    outputs = np.argmax(out, axis=1)
    return np.sum(outputs == labels)

