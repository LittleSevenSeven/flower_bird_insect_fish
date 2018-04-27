#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 21 17:07:22 2017

@author: ycy
"""

#LeNet

import numpy as np
import keras
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
#from keras.layers import Conv2D, MaxPooling2D
from keras.layers.convolutional import Convolution2D, MaxPooling2D #卷积
from keras.optimizers import SGD, Adadelta, Adagrad,Adam #优化算法
from keras.layers.advanced_activations import PReLU
from keras.utils import np_utils, generic_utils


#-------------------------------pretreatment 预处理-----------------------------------
X_train=np.load('/Users/ycy/Desktop/image_train.npy')
X_train=X_train/255.0
Y_train=np.load('/Users/ycy/Desktop/label_train.npy')
Y_train = keras.utils.to_categorical(Y_train,num_classes=4) # convert class vectors to binary class matrices

X_test=np.load('/Users/ycy/Desktop/image_test.npy')
X_test=X_test/255.0
Y_test=np.load('/Users/ycy/Desktop/label_test.npy')
Y_test = keras.utils.to_categorical(Y_test,num_classes=4)


#---------------------------build the neural net 建模型----------------------------
model = Sequential()
 
#第一个卷积层，4个卷积核，每个卷积核大小5*5。1表示输入的图片的通道,灰度图为1通道。
#border_mode可以是valid或者full，具体看这里说明：http://deeplearning.net/software/theano/library/tensor/nnet/conv.html#theano.tensor.nnet.conv.conv2d
#激活函数用tanh
#你还可以在model.add(Activation('tanh'))后加上dropout的技巧: model.add(Dropout(0.5))
model.add(Convolution2D(8, 5, 5, border_mode='valid', input_shape=(64,64,3)))
model.add(Activation('relu'))
model.add(Dropout(0.25))

#第二个卷积层，8个卷积核，每个卷积核大小3*3。4表示输入的特征图个数，等于上一层的卷积核个数
#激活函数用tanh
#采用maxpooling，poolsize为(2,2)
model.add(Convolution2D(32, 3, 3, border_mode='valid'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
 
#第三个卷积层，16个卷积核，每个卷积核大小3*3
#激活函数用tanh
#采用maxpooling，poolsize为(2,2)
model.add(Convolution2D(16, 3, 3, border_mode='valid'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
 
#全连接层，先将前一层输出的二维特征图flatten为一维的。
#Dense就是隐藏层。16就是上一层输出的特征图个数。4是根据每个卷积层计算出来的：(28-5+1)得到24,(24-3+1)/2得到11，(11-3+1)/2得到4
#全连接有128个神经元节点,初始化方式为normal
model.add(Flatten())
model.add(Dense(128, init='normal'))
model.add(Dropout(0.5))
model.add(Activation('relu'))
 
 
#Softmax分类，输出是4类别
model.add(Dense(4, init='normal'))
model.add(Activation('softmax'))

#-------------------------compile the model 编译模型-----------------------------
#使用SGD + momentum
#model.compile里的参数loss就是损失函数(目标函数)
sgd = SGD(lr=0.05, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

#--------------------------train the model 训练模型-----------------------------

#调用fit方法，就是一个训练过程. 训练的epoch数设为10，batch_size为100．
#数据经过随机打乱shuffle=True。verbose=1，训练过程中输出的信息，0、1、2三种方式都可以，无关紧要。show_accuracy=True，训练时每一个epoch都输出accuracy。
#validation_split=0.2，将20%的数据作为验证集。
model.fit(X_train, Y_train, batch_size=5, nb_epoch=100,shuffle=True,verbose=1,validation_data=(X_test, Y_test))



#--------------------------test the model 测试模型------------------------------
score = model.evaluate(X_test, Y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])