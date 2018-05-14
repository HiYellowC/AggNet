# -*- coding: utf-8 -*-
# @Time    : 2018/5/9 下午4:38
# @File    : crowdsourcing.py

import numpy as np
import random
import pickle
import os

def get_crowdsourcing_y(num_anchors, true_y):

    if os.path.exists('./crowdsourcing_y.txt'):
        with open('./crowdsourcing_y.txt', 'rb') as f:
            count_alpha, count_beta, crowdsourcing_y = pickle.load(f)
        print('load success')
        return count_alpha, count_beta, crowdsourcing_y

    count_alpha = []
    count_beta = []
    for i in range(num_anchors):
        count_alpha.append(random.uniform(0.5, 1))
        count_beta.append(random.uniform(0.5, 1))

    print('count_alpha', count_alpha)
    print('count_beta', count_beta)

    crowdsourcing_y = np.zeros((num_anchors, len(true_y)))
    for idx_anchors in range(num_anchors):
        for i in range(len(true_y)):
            p = random.random()
            if (true_y[i] == 1 and p < count_alpha[idx_anchors]) or (true_y[i] == 0 and p > count_beta[idx_anchors]):
                crowdsourcing_y[idx_anchors, i] = 1
            else:
                crowdsourcing_y[idx_anchors, i] = 0

    with open('./crowdsourcing_y.txt', 'wb') as f:
        pickle.dump([count_alpha, count_beta, crowdsourcing_y], f)
        print('save success')

    return count_alpha, count_beta, crowdsourcing_y