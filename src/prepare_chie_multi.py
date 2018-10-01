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
            qid_que_ans = pair.split('\t')

            qid = qid_que_ans[0]
            que_str = qid_que_ans[1]
            ans_str = qid_que_ans[2]

            #preprocessing
            que_list = preprocessing(que_str)
            ans_list = preprocessing(ans_str)

            #padding
            que_list = padding(que_list)
            ans_list = padding(ans_list)

            #join
            que_str = '_'.join(que_list) + '_'
            ans_str = '_'.join(ans_list) + '_'

            write_line(w, qid, que_str, str(ID), ans_str)


def write_line(w, tok1, tok2, tok3, tok4):
    output_list = []
    output_list.append(tok1)
    output_list.append(tok2)
    output_list.append(tok3)
    output_list.append(tok4)
    output_str = ' '.join(output_list)

    w.write(output_str)
    w.write('\n')


def make_id_list(line):
    '''
    line = [(1, 'q1', 1, 'a1'), 
            (1, 'q1', 2, 'a2'),
            (2, 'q2', 3, 'a3'),
            (3, 'q3', 4, 'a4'),
            (3, 'q3', 5, 'a5')]

    qa_list = [[qid1, aid1, aid2],
               [qid2, aid3],
               [qid3, aid4, aid5]]
    [[1,1,2],
     [2,3],
     [3,4,5]]
    '''
    qa_list = []
    for i, (qid, que, aid, ans) in enumerate(line):
        new_qid = qid
        if i == 0: #初めて
            pair_list = [] #pair
            pair_list.append(new_qid) #pair
            pair_list.append(aid) #pair

        elif pre_qid == new_qid:
            pair_list.append(aid) #pair

        elif pre_qid != new_qid: #新しく質問が変化。今までのpair_listをqa_listに追加して、pair_listは初期化
            qa_list.append(pair_list) #qa

            pair_list = [] #pair
            pair_list.append(new_qid) #pair
            pair_list.append(aid) #pair
        pre_qid = qid

    qa_list.append(pair_list)
    return qa_list

def make_id_content_dict(pairs):
    que_id_content_dict = {}
    ans_id_content_dict = {}
    for qid, que, aid, ans in pairs:
        que_id_content_dict[str(qid)] = que
        ans_id_content_dict[str(aid)] = ans
    return que_id_content_dict, ans_id_content_dict 

def make_eval_data(input_file, output_file, type):
    with open(input_file, 'r') as r, open(output_file, 'w') as w:
        pairs = [line.split() for line in r.readlines()]
        qa_id_list = make_id_list(pairs)
        que_id_content_dict, ans_id_content_dict = make_id_content_dict(pairs)
        if type == 'dev':
            eval_index = 1000
        elif type == 'test':
            eval_index = 1800
        for QID, qid_aid_list in enumerate(qa_id_list[:eval_index]):
            que_content = que_id_content_dict[str(qid_aid_list[0])]

            count_1000 = 0
            for aid in qid_aid_list[1:]: # iter aid
                ans_content = ans_id_content_dict[str(aid)]

                write_line(w, '1', 'qid:' + str(QID) + '.' + str(count_1000).zfill(3), que_content, ans_content)
                count_1000 += 2

            # make fake data
            size_fake = 500 - (len(qid_aid_list) - 1)

            # Make Fake Data
            alist = load_alist(pairs)

            for fake_aid, fake_a_seq in random.sample(alist, 500):
                if count_1000 < 1000:
                    if str(fake_aid) not in qid_aid_list[1:]:
                        write_line(w, '0', 'qid:' + str(QID) + '.' + str(count_1000).zfill(3), que_content, fake_a_seq)
                        count_1000 += 2
                else:
                    break


def make_train_data(input_file, output_file):
    with open(input_file, 'r') as r, open(output_file, 'w') as w:
        pairs = [line.split() for line in r.readlines()]
        index = 12877
        for que_id, q_seq, ans_id, a_seq in pairs[:index]:
            write_line(w, '1', 'qid:0.0', q_seq, a_seq)


def load_alist(pairs):
    alist = []
    for qid, q_seq, aid, a_seq in pairs:
        alist.append((aid, a_seq))
    return alist


def divide_data(file_qa, file_train, file_dev, file_test1, file_test2):
    size_train = 12887 * 2
    size_dev = 1000 * 2
    size_test1 = 1800 * 2
    size_test2 = 1800 * 2

    index_train = size_train
    index_dev = size_train + size_dev
    index_test1 = size_train + size_dev + size_test1
    index_test2 = size_train + size_dev + size_test1 + size_test2

    with open(file_qa, 'r') as r:
        lines = [line for line in r.readlines()]
    with open(file_train, 'w') as train_w, open(file_dev, 'w') as dev_w, open(file_test1, 'w') as test1_w, open(file_test2, 'w') as test2_w:
        for i, line in enumerate(lines):

            if i < index_train:
                train_w.write(line)
                #train_w.write('\n')

            elif i < index_dev:
                dev_w.write(line)
                #dev_w.write('\n')

            elif i < index_test1:
                test1_w.write(line)
                #test1_w.write('\n')

            elif i < index_test2:
                test2_w.write(line)
                #test2_w.write('\n')


if __name__ == '__main__':
    max_length = 200
    #print('making qa pairs')
    #make_qa_pairs('./dataset/one_que_content_multi_ans_content_2017', './chieQA', max_length)

    #print('dividing data')
    #divide_data('./chieQA', './chieTrain', './chieDev', './chieTest1', './chieTest2')

    print('making train data')
    make_train_data('./chieTrain', './train')

    #print('making dev data')
    #make_eval_data('./chieDev', './dev', 'dev')
    
    print('making test1 data')
    make_eval_data('./chieTest1', './test1', 'test')

    print('making test2 data')
    make_eval_data('./chieTest2', './test2', 'test')
    
