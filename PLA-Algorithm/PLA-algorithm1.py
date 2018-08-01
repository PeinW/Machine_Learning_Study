#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2017/11/18 20:47
# @Author  : Pein Wang
# @Email   : lihuazhang@live.com
# @File    : PLA_algorithm.py
# @Software:
"""
Description :
    PLA - POCKET algorithm
"""
import numpy as np
import time
import random

TRAIN_FILE = r'train_data.csv'
TARGET_LOSS = 0
ITERATION = 20 # number of iteration


class PLA_algorithm(object):

    def __init__(self, file_name):
        data_group = np.genfromtxt(file_name, delimiter=',', dtype=np.str)
        self.train_data = data_group[1:, ].astype(np.float)
        self.simple_num = self.train_data.shape[0]
        self.w_0 = np.zeros(self.train_data.shape[1]-1, np.float)

    def sign(self, x_t, weight):
        w_temp = np.dot(weight, x_t)
        if w_temp >= 0:
            return 1
        else:
            return -1

    def judge_data(self, simple, weight):

        label = simple[-1]
        data_t = simple[0:-1]
        sign_result = self.sign(data_t, weight)
        if sign_result == label:
            flag = 'match'
        else:
            flag = 'mistake'

        return flag

    def fresh_w(self, simple, weight):

        label = simple[-1]
        data_t = simple[0:-1]
        weight_new = weight + label * data_t

        return weight_new

    def count_loss(self, weight):
        mistake_sum = 0
        for simple in self.train_data:
            judge_result = self.judge_data(simple, weight)
            if judge_result == 'mistake':
                mistake_sum += 1

        loss_rate = mistake_sum / self.simple_num
        return loss_rate

    @property
    def get_weight(self):
        return self.w_0

    @property
    def get_data(self):
        return self.train_data

    def __str__(self):
        return 'This is PLA method'


my_PLA = PLA_algorithm(TRAIN_FILE)
train_data = my_PLA.get_data
weight = my_PLA.get_weight

loss_rate = my_PLA.count_loss(weight)
loss_rate_best = loss_rate

times = 0
started = time.time()
print('-'*30)
print('PLA Pocket algorithm')
print('The First Loss Rate is %s' % loss_rate)

while times <= ITERATION:
    if loss_rate_best == 0:
        print('We have got the Golden Perception')
        break
    simple = random.choice(train_data)

    if my_PLA.judge_data(simple, weight) == 'mistake':
        weight = my_PLA.fresh_w(simple, weight)
        loss_rate = my_PLA.count_loss(weight)
        if (loss_rate <= loss_rate_best):
            loss_rate_best = loss_rate
            times += 1
            print('-'*30)
            print('Refresh Loss Rate and Weight')
            print('Iteration : %s' % times)
            print('The Best Loss Rate is %s' % loss_rate)
            print('The Weight = %s' % weight)
            if times == ITERATION:
                break

ending = time.time()
during_time = ending - started
print('-' * 50)
print('Pocket learning algorithm is over')
print('Train time is %f s' % during_time)
print('-'*50)
