#!/usr/bin/python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import csv

current_path = os.getcwd()
lib_path = os.path.join(current_path, 'data')
data_path = os.path.join(lib_path, 'train.csv')

train_data = []
x_data = []
y_data = []
# load original data
with open(data_path, 'r', encoding='big5') as fh:
    clumps = csv.reader(fh, delimiter = ',')
    for data in clumps:
        train_data.append(data)
    fh.close()
train_data = np.array(train_data)
train_data = train_data[1:,]
# stripping PM2.5 data
location = np.where(train_data == 'PM2.5')
pm25_data = train_data[location[0],]
pm25_data = pm25_data[:,3:]
data_size = pm25_data.shape[0]*pm25_data.shape[1]
# sorting the data as line
pm25_data = pm25_data.reshape([1, data_size])
pm25_data = pm25_data.astype(np.float)
# split x_data and y_data ---- Redundant
# x_data = PM2.5 in 0~9st
# y_data = PM2.5 at 10st
for i in range(data_size):
    if (i+10) < data_size:
        x_data.append(pm25_data[0, i:(i+9)])
        y_data.append(pm25_data[0, i+10])
x_data = np.array(x_data, np.float)
y_data = np.array(y_data, np.float)
# h(x) = weight * x_data + bias
weight = np.ones(9)
result = np.dot(weight, x_data[0,])
