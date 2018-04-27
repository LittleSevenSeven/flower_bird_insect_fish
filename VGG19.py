#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 27 21:50:55 2017

@author: ycy
"""

from keras.optimizers import Adam
from keras import models 
from keras import layers 
from keras import optimizers
import time
import json #json模块可以直接处理简单数据类型（string、unicode、int、float、list、tuple、dict）
import numpy as np
from keras.applications import imagenet_utils, VGG19
from keras.applications.vgg19 import decode_predictions, preprocess_input
from keras.layers import Dense, Dropout, Flatten, Input
from keras.models import load_model, Model, Sequential
from keras.optimizers import Adam
from keras.preprocessing import image
from keras.utils import np_utils #keras.utils神经网络可视化模块
from keras.callbacks import ReduceLROnPlateau #keras.callbacks回掉函数 #Reduce learning rate when a metric（度量标准） has stopped improving.
from keras.preprocessing.image import ImageDataGenerator
import h5py.h5f
from keras.applications.vgg19 import VGG19



height = 64
width = 64
conv_base = VGG19(weights='imagenet',include_top=False,input_shape=(height, width, 3),classes=4)
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
std = np.std(x_train, axis=(0, 1, 2, 3))
x_train = (x_train-mean)/(std+1e-7) # 1e相当于10，防止除以0
x_test = (x_test-mean)/(std+1e-7)


# Set a learning rate annealer
# 使用LearningRateScheduler，用来在训练停滞不前的时候动态降低学习率
# 使用ReduceLROnPlateau，在训练进入平台期的时候动态调节学习率 栗子eg. http://blog.csdn.net/tsyccnh/article/details/78865167
# 当评价指标不再提升时，减少学习率
# de该回掉函数检测指标的情况，如果在penitence个epoch中看不到模型性能提升，则减少学习率
# ReduceLROnPlateau参数详解 http://www.360doc.com/content/17/0126/22/40028542_624945104.shtml
# monitor 被监测的量
# factor 每次减少学习率的因子，学习率将以lr = lr * factor的形式被减少
# patience 当patience个epoch过去而模型性能不提升时，学习率减少的动作会被触发
# min_lr 学习率的下限
# verbose: int. 0: quiet, 1: update messages. ???
learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', 
                                            patience=3, 
                                            verbose=1, 
                                            factor=0.5, 
                                            min_lr=0.00001)


# -------- optimizer setting -------- #
sgd = optimizers.SGD(lr=.0001, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])


# 训练模型
#history=model.fit(x_train,y_train,batch_size=64,epochs=20,validation_data=(x_test,y_test))
# ImageDataGenerator 用于实时数据提升 http://blog.sina.com.cn/s/blog_62cd91d50102wiwq.html
# ImageDataGenerator参数详解 http://blog.csdn.net/u012969412/article/details/76796020
# horizontal_flip 进行随机水平翻转
# width_shift_range 浮点数，图片宽度的某个比例，数据提升时图片随机水平偏移的幅度。
# fill_mode ‘constant’，‘nearest’，‘reflect’或‘wrap’之一，当进行变换时超出边界的点将根据本参数给定的方法进行处理
# cval：浮点数或整数，当fill_mode=constant时，指定要向超出边界的点填充的值。
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