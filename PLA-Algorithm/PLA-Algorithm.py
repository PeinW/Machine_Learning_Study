#!/usr/bin/python3
# -*- coding: utf-8 -*-

import csv
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

current_path = os.getcwd()
data_path = os.path.join(current_path, 'data')
sys.path.append(data_path)


data = []

# 每一個維度儲存一種污染物的資訊
for i in range(18):
    data.append([])

n_row = 0
text = open(data_path+'\\train.csv', 'r', encoding='big5')
row = csv.reader(text, delimiter=",")
for r in row:
    # 第0列沒有資訊

    if n_row != 0:
        # 每一列只有第3-27格有值(1天內24小時的數值)
        for i in range(3,27):
            if r[i] != "NR":
                data[(n_row-1) % 18].append(float(r[i]))
            else:
                data[(n_row-1) % 18].append(float(0))
    n_row = n_row+1

text.close()
data = np.array(data)
print(data.shape)