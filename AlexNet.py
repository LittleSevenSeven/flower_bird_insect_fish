#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 29 00:06:58 2017

@author: ycy
"""

from keras.optimizers import Adam
from keras import models 
from keras import layers 
from keras import optimizers
import time
#import json
import numpy as np
#from keras.applications import imagenet_utils
from keras.layers import Dense, Dropout, Flatten,Conv2D,MaxPooling2D
from keras.models import  Model, Sequential
#from keras.preprocessing import image
#from keras.utils.np_utils import to_categorical
from keras.callbacks import ReduceLROnPlateau
#import h5py.h5f
import keras
from keras.preprocessing.image import ImageDataGenerator


height = 64
width = 64

# load data
x_train=np.load('/Users/ycy/Desktop/image_train.npy')
y_train=np.load('/Users/ycy/Desktop/label_train.npy')
#for i in range(400):
#    y_train[i,:]=y_train[i,:]-1
y_train = keras.utils.to_categorical(y_train,num_classes=4) # convert class vectors to binary class matrices
    
x_test=np.load('/Users/ycy/Desktop/image_test.npy')
y_test=np.load('/Users/ycy/Desktop/label_test.npy')
#for i in range(400):
#    y_test[i,:]=y_test[i,:]-1
y_test = keras.utils.to_categorical(y_test,num_classes=4)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')    

mean = np.mean(x_train,axis=(0,1,2,3))
std = np.std(x_train, axis=(0, 1, 2, 3))
x_train = (x_train-mean)/(std+1e-7)
x_test = (x_test-mean)/(std+1e-7)
y_test=np.array(y_test)
Y_test = keras.utils.to_categorical(y_test,num_classes=4)

learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', 
                                            patience=10, 
                                            verbose=1, 
                                            factor=0.1, 
                                            min_lr=0.00001)

model = Sequential()  
model.add(Conv2D(96,(11,11),strides=(4,4),input_shape=(64,64,3),padding='valid',activation='relu',kernel_initializer='uniform'))  
model.add(MaxPooling2D(pool_size=(3,3),strides=(2,2)))  
model.add(Conv2D(256,(5,5),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))  
model.add(MaxPooling2D(pool_size=(3,3),strides=(2,2)))  
model.add(Conv2D(384,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))  
model.add(Conv2D(384,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))  
model.add(Conv2D(256,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))  
#model.add(MaxPooling2D(pool_size=(3,3),strides=(2,2)))  
model.add(Flatten())  
model.add(Dense(4096,activation='relu'))  
model.add(Dropout(0.5))  
model.add(Dense(4096,activation='relu'))  
model.add(Dropout(0.5))  
model.add(Dense(4,activation='softmax'))  
#adam=keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
adam=keras.optimizers.Adamax(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
model.compile(loss='categorical_crossentropy',optimizer=adam,metrics=['accuracy'])  
model.summary() 
datagen = ImageDataGenerator(horizontal_flip=True,
        width_shift_range=0.125,height_shift_range=0.125,fill_mode='constant',cval=0.)

datagen.fit(x_train)

model.fit_generator(datagen.flow(x_train, y_train,batch_size=50),
                    steps_per_epoch=8,
                    epochs=70,
                    callbacks=[learning_rate_reduction],
                    validation_data=(x_test, y_test))