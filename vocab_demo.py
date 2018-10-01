import numpy as np
import random
import math
import collections


# Attention: vocab from not only train but also dev and test.
def build_vocab():
    code = int(0)
    vocab = {}
    vocab['UNKNOWN'] = code
    code += 1
    #for filename in ['chieQA/train','chieQA/test1','chieQA/test2','chieQA/dev']:
    for filename in ['chieQA/dev']:
        for line in open(filename):
            items = line.strip().split(' ')
            for i in range(2, 3):
                words = items[i].split('_')
                for word in words:
                    if not word in vocab:
                        vocab[word] = code
                        code += 1
    return vocab


def make_dic():
    # 頻度順にソートしてidをふる
    words = []
    vocab = {}
    for filename in ['chieQA/train','chieQA/test1','chieQA/test2','chieQA/dev']:
        for line in open(filename):
            items = line.strip().split(' ')
            if int(items[0]) == 1:
                for i in range(2, 3):
                    words.extend(items[i].split('_'))

    counter = collections.Counter()
    counter.update(words)
    cnt = 0
    for word, count in counter.most_common():
        # 出現回数15回以上の単語のみ辞書に追加
        if count >= 15:
            vocab[word] = cnt
            cnt += 1
    vocab[u'<unk>'] = len(vocab)
    return vocab


def count_len(seq_str):
    seq_list = [word for word in seq_str.split('_') if word != '<a>']
    seq_len = len(seq_list) - 1
    return seq_len


def avg_length():
    for filename in ['insuranceQA/train']:
        que_len = 0
        ans_len = 0
        for cnt, line in enumerate(open(filename)):
            items = line.strip().split(' ')
            que_len += count_len(items[2])
            ans_len += count_len(items[3])
        avg_que = que_len / cnt
        avg_ans = ans_len / cnt

    return avg_que, avg_ans


#vocab = make_dic()
#print(vocab)
avg_que, avg_ans = avg_length()
print(avg_que)
print(avg_ans)
