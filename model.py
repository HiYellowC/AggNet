# -*- coding: utf-8 -*-
# @Time    : 2018/5/9 上午11:06
# @File    : model.py

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation
from keras.layers import Flatten, Dense, Dropout, BatchNormalization
from keras.optimizers import Adam

def get_model(image_target_size):

    # model = Sequential()
    # model.add(Conv2D(128, (3, 3), padding='same', input_shape=(image_target_size, image_target_size, 3)))
    # model.add(BatchNormalization())
    # model.add(Activation('relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    #
    # model.add(Conv2D(128, (3, 3), padding='same'))
    # model.add(BatchNormalization())
    # model.add(Activation('relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    #
    # model.add(Conv2D(256, (3, 3), padding='same'))
    # model.add(BatchNormalization())
    # model.add(Activation('relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    #
    # model.add(Conv2D(256, (3, 3), padding='same'))
    # model.add(BatchNormalization())
    # model.add(Activation('relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))

    model = Sequential()
    model.add(Conv2D(16, (3, 3), padding='same', input_shape=(image_target_size, image_target_size, 3)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(16, (3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(16, (3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(100))
    model.add(Activation('relu'))
    model.add(Dropout(0.3))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    optimizer = Adam(0.001, 0.9, 0.999, None)
    model.compile(loss='binary_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])
    return model
