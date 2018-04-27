#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  3 23:18:07 2018

@author: ycy
"""

from keras.optimizers import Adam
from keras import models 
from keras import layers 
from keras import optimizers
import time
import json #json模块可以直接处理简单数据类型（string、unicode、int、float、list、tuple、dict）
import numpy as np
from keras.applications import imagenet_utils, VGG16
from keras.applications.vgg16 import decode_predictions, preprocess_input
from keras.layers import Dense, Dropout, Flatten, Input
from keras.models import load_model, Model, Sequential
from keras.optimizers import Adam
from keras.preprocessing import image
from keras.utils import np_utils #keras.utils神经网络可视化模块
from keras.callbacks import ReduceLROnPlateau #keras.callbacks回掉函数 #Reduce learning rate when a metric（度量标准） has stopped improving.
from keras.preprocessing.image import ImageDataGenerator
import h5py.h5f
from keras.applications.vgg16 import VGG16



height = 64
width = 64
conv_base = VGG16(weights='imagenet',include_top=False,input_shape=(height, width, 3),classes=4)
#conv_base = VGG_16()
start=time.clock()
model = models.Sequential() 
model.add(conv_base) 
model.add(layers.Flatten()) 
model.add(layers.Dense(256, activation='relu')) 
model.add(Dropout(0.5))
model.add(layers.Dense(4, activation='softmax'))
conv_base.trainable = True

# load data
x_train=np.load('/Users/ycy/Desktop/image_train.npy')
y_train=np.load('/Users/ycy/Desktop/label_train.npy')
y_train = np_utils.to_categorical(y_train,num_classes=4) # convert class vectors to binary class matrices
    
x_test=np.load('/Users/ycy/Desktop/image_test.npy')
y_test=np.load('/Users/ycy/Desktop/label_test.npy')
y_test = np_utils.to_categorical(y_test,num_classes=4)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')    
#    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
#    y_train = keras.utils.to_categorical(y_train, num_classes)
#    y_test = keras.utils.to_categorical(y_test, num_classes)
#    x_train = x_train.astype('float32')
#    x_test = x_test.astype('float32')
    
    # data preprocessing  [raw - mean / std]
mean = np.mean(x_train,axis=(0,1,2,3))
std = np.std(x_train, axis=(0,1,2,3))
x_train = (x_train-mean)/(std+1e-7) # 1e相当于10，防止除以0
x_test = (x_test-mean)/(std+1e-7)


# Set a learning rate annealer
learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', 
                                            patience=3, 
                                            verbose=1, 
                                            factor=0.5, 
                                            min_lr=0.00001)


# -------- optimizer setting -------- #
sgd = optimizers.SGD(lr=.0001, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])


# 训练模型
# 数据增强
datagen = ImageDataGenerator(horizontal_flip=True,
        width_shift_range=0.125,height_shift_range=0.125,fill_mode='constant',cval=0.)

datagen.fit(x_train)

# fit_generator 用于从Python生成器中训练网络 http://blog.sina.com.cn/s/blog_62cd91d50102wiwq.html
model.fit_generator(datagen.flow(x_train, y_train,batch_size=64),
                    steps_per_epoch=6,
                    epochs=50,
                    callbacks=[learning_rate_reduction],
                    validation_data=(x_test, y_test))