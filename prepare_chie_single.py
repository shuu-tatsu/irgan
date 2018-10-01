#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
import random

def padding(sent_list):
    while True:
        if len(sent_list) > max_length:
            break
        else:
            sent_list.append('<a>')
    return sent_list


def preprocessing(seq_str):
    #'_' は単語の境界にする為、余計なものは'-'に変換しておく
    #seq_str.replace('_', '-')

    seq_list = seq_str.strip().split()

    #ノイズ除去
    resub_list = []
    for word_str in seq_list:
        resub_word = re.sub('[^あ-んア-ン\u4E00-\u9FD0]', '', word_str)
        if len(resub_word) > 0:
            resub_list.append(resub_word)

    return resub_list


def make_qa_pairs(input_file, output_file, max_length):
    with open(input_file, 'r') as r, open(output_file, 'w') as w:
        pairs = [line for line in r.readlines()]
        for ID, pair in enumerate(pairs):
            que_ans = pair.split('\t')

            que_str = que_ans[0]
            ans_str = que_ans[1]

            #preprocessing
            que_list = preprocessing(que_str)
            ans_list = preprocessing(ans_str)

            #padding
            que_list = padding(que_list)
            ans_list = padding(ans_list)

            #join
            que_str = '_'.join(que_list) + '_'
            ans_str = '_'.join(ans_list) + '_'

            write_line(w, str(ID), que_str, str(ID), ans_str)


def write_line(w, label, ID, q_seq, a_seq):
    output_list = []
    output_list.append(label)
    output_list.append(ID)
    output_list.append(q_seq)
    output_list.append(a_seq)
    output_str = ' '.join(output_list)

    w.write(output_str)
    w.write('\n')


def make_eval_data(input_file, output_file):
    with open(input_file, 'r') as r, open(output_file, 'w') as w:
        pairs = [line.split() for line in r.readlines()]
        for ID, (qid, q_seq, aid, a_seq) in enumerate(pairs):
            write_line(w, '1', 'qid:' + str(ID) + '.0', q_seq, a_seq)

            # Make Fake Data
            num_write_line = 1 #正解QAペアは読み取り済み。あとは499のフェイクQAペアを読みとる。
            alist = load_alist(pairs)

            for fake_aid, fake_a_seq in random.sample(alist, 500):
                if num_write_line < 500:
                    if aid != fake_aid:
                        count_id_str = str(num_write_line * 2)
                        count_id_str = count_id_str.zfill(3)
                        write_line(w, '0', 'qid:' + str(ID) + '.' + count_id_str, q_seq, fake_a_seq)
                        num_write_line += 1
                else:
                    break


def make_train_data(input_file, output_file):
    with open(input_file, 'r') as r, open(output_file, 'w') as w:
        pairs = [line.split() for line in r.readlines()]
        for qid, q_seq, aid, a_seq in pairs:
            write_line(w, '1', 'qid:0.0', q_seq, a_seq)


def load_alist(pairs):
    alist = []
    for qid, q_seq, aid, a_seq in pairs:
        alist.append((aid, a_seq))
    return alist


if __name__ == '__main__':
    max_length = 200
    #make_qa_pairs('dataset/q_a_1to1_pairs_2017', './chieQA', max_length)
    make_train_data('./chieTrain', './train')
    make_eval_data('./chieDev', './dev')
    make_eval_data('./chieTest1', './test1')
    make_eval_data('./chieTest2', './test2')
