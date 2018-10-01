#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
sys.path.append('./')
from operator import itemgetter

file_path_read = './model_name.out'
file_dis_precision = './result/dis_precision.out'

epoch = 0

d_list = []

with open(file_path_read, 'r') as r, open(file_dis_precision, 'w') as dw:
    lines = r.readlines()
    for line in lines:
        line_split = line.split()
        for model in line_split:
            if 'Dis' in model:
                tok = model.split('-')
                val = tok[0].replace('Dis', '')
                time = tok[1].replace('.model', '')
                d_list.append([val, time])

    d_list.sort(key=itemgetter(1))

    for val, time in d_list:
        dw.write(val)
        dw.write('\n')

