# -*- coding: utf-8 -*-
# @Time    : 2018/5/9 上午11:07
# @File    : main.py

import model
import glob
import random
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
import numpy as np
import os
import tqdm
import cv2
import crowdsourcing

# 指定显卡
os.environ['CUDA_VISIBLE_DEVICES'] = '1'


EGRET_IMG_PATH = './birds/egret'
MANDARIN_IMG_PATH = './birds/mandarin'

egret_data = glob.glob(EGRET_IMG_PATH + '/*.*')
mandarin_data = glob.glob(MANDARIN_IMG_PATH + '/*.*')

train_data = egret_data[:80] + mandarin_data[:80]
validation_data = egret_data[80:] + mandarin_data[80:]

num_epochs = 100
image_target_size = 33
num_anchors = 10
# 初始置信度
init_believe = 0.9
alpha = [init_believe] * num_anchors
beta = [init_believe] * num_anchors
# 获取群治标注集
count_alpha, count_beta, crowdsourcing_y = crowdsourcing.get_crowdsourcing_y(num_anchors, [0] * 80 + [1] * 80)
print('crowdsourcing_y', crowdsourcing_y)
# 训练集 图片与序号的映射
dict = {}
for i, data in enumerate(train_data):
    dict[data] = i

validation_x = []
for data in validation_data:
    try:
        img = image.load_img(data, target_size=(image_target_size, image_target_size))
    except OSError or AttributeError:
        img = np.zeros(shape=(image_target_size, image_target_size, 3))
        continue

    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    validation_x.extend(x)

validation_x = np.asarray(validation_x)

def get_avg_error(x1, x2):
    sum = 0.0
    for xx1, xx2 in zip(x1, x2):
        sum += abs(xx1 - xx2)
    return sum / len(x1)

model = model.get_model(image_target_size)

init_sum = 100.0
sum_ui = init_sum
sum_neg_ui = init_sum
sum_ui_yi = [init_sum * init_believe] * num_anchors
sum_neg_ui_neg_yi = [init_sum * init_believe] * num_anchors

print('----init alphf avg error = {}'.format(get_avg_error(count_alpha, alpha)), '-----------')
print('----init beta avg error = {}'.format(get_avg_error(count_beta, beta)), '-----------')

for epoch_num in range(num_epochs):

    random.shuffle(train_data)

    # 错误的ui数
    cont = 0
    for data in tqdm.tqdm(train_data):

        # x_img = cv2.imread(data)
        # x_img = cv2.resize(x_img, (image_target_size, image_target_size), interpolation=cv2.INTER_CUBIC)
        # x_img = x_img[:, :, (2, 1, 0)]  # BGR -> RGB
        # x_img = x_img.astype(np.float32)
        #
        # x_img = np.transpose(x_img, (2, 0, 1))
        # x_img = np.expand_dims(x_img, axis=0)
        #
        # x_img = np.transpose(x_img, (0, 2, 3, 1))

        idx_data = dict[data]

        try:
            img = image.load_img(data, target_size=(image_target_size, image_target_size))
        except OSError or AttributeError:
            img = np.zeros(shape=(image_target_size, image_target_size, 3))
            continue

        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)

        p, filename = os.path.split(data)
        #
        # if filename[:3] == 'egr':
        #     y = [0]
        # else:
        #     y = [1]

        x = np.asarray(x)
        # y = np.asarray(y)

        # 计算 ui
        pi = model.predict_on_batch(x)
        ai = 1.0
        bi = 1.0
        for idx_anchors in range(num_anchors):
            ai *= ((alpha[idx_anchors] ** crowdsourcing_y[idx_anchors, idx_data]) * ((1 - alpha[idx_anchors]) ** (1 - crowdsourcing_y[idx_anchors, idx_data])))
            bi *= (beta[idx_anchors] ** (1 - crowdsourcing_y[idx_anchors, idx_data]) * (1 - beta[idx_anchors]) ** crowdsourcing_y[idx_anchors, idx_data])
        ui = (ai * pi) / (ai * pi + bi * (1 - pi))

        # 计算GT
        if ui > 0.5:
            y = [1]
        else:
            y = [0]
        y = np.asarray(y)

        if (filename[:3] == 'egr' and y[0] == 1) or (filename[:3] != 'egr' and y[0] == 0):
            cont += 1

        # BP
        model.train_on_batch(x, y)
        # 更新alpha beta
        sum_ui += ui
        sum_neg_ui += (1 - ui)
        for idx_anchors in range(num_anchors):
            sum_ui_yi[idx_anchors] += ui * crowdsourcing_y[idx_anchors, idx_data]
            sum_neg_ui_neg_yi[idx_anchors] += (1 - ui) * (1 - crowdsourcing_y[idx_anchors, idx_data])
            alpha[idx_anchors] = sum_ui_yi[idx_anchors] / sum_ui
            beta[idx_anchors] = sum_neg_ui_neg_yi[idx_anchors] / sum_neg_ui

    print('----epoch ', epoch_num, 'missed lable cont = {}'.format(cont), '-----------')
    pred_y = model.predict_on_batch(validation_x)
    true_y = [0] * 20 + [1] * 20

    cont = 0
    for p_y, t_y in zip(pred_y, true_y):

        if (p_y < 0.5 and t_y == 0) or (p_y >= 0.5 and t_y ==1):
            cont += 1

    print('----epoch ', epoch_num, 'acc = {}'.format(cont / 40.0), '-----------')

    print('----epoch ', epoch_num, 'alphf avg error = {}'.format(get_avg_error(count_alpha, alpha)), '-----------')
    print('----epoch ', epoch_num, 'beta avg error = {}'.format(get_avg_error(count_beta, beta)), '-----------')