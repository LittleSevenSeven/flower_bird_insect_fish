#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 20 11:37:55 2017

@author: ycy
"""

import numpy as np
import PIL
import PIL.Image as im
import matplotlib.pyplot as plt
import random



# 在to_npy5保存灰度图片，to_npy7保存RGB图片


#------------------------------选4组随机数-----------------------------------
random.seed(10)
x = range(1,201)
y = list(x)
random.shuffle(y)
print(y) #y是被打乱的1-200
m1 = [] #m-train
n1 = [] #n-test
for k in range(100):
    m1.append(y[k]) #m[k]是被打乱的1-200的前100个数
    n1.append(y[k+100]) #n[k]是被打乱的1-200的后100个数
print(m1)
print(n1)


random.seed(9)
y = list(x)
random.shuffle(y)
print(y)
m2 = [] 
n2 = [] 
for k in range(100):
    m2.append(y[k]) 
    n2.append(y[k+100]) 
print(m2)
print(n2)


random.seed(8)
y = list(x)
random.shuffle(y)
print(y)
m3 = [] 
n3 = [] 
for k in range(100):
    m3.append(y[k]) 
    n3.append(y[k+100]) 
print(m3)
print(n3)


random.seed(7)
y = list(x)
random.shuffle(y)
print(y)
m4 = [] 
n4 = [] 
for k in range(100):
    m4.append(y[k]) 
    n4.append(y[k+100]) 
print(m4)
print(n4)


#----------------------------------分8个文件存npy-------------------------------------
## image_flower_train
array_flower_train = np.zeros(((100,64,64,3)))
j = 0
for i in m1: #左闭右开
    img = im.open('/Users/ycy/Desktop/flower_frommysql/'+str(i)+'.jpg').convert("RGB") #L为灰度图，RGB为真彩色，CMYK为pre-press图像
    out = img.resize((64,64))
    arr = np.array(out)
    np.save('/Users/ycy/Desktop/array_train&test/arr/flower/train/'+str(i),arr)
    array_flower_train[j] = arr
    j = j + 1    
np.save('/Users/ycy/Desktop/array_train&test/image_flower_train',array_flower_train)

## image_flower_test
array_flower_test = np.zeros(((100,64,64,3)))
j = 0
for i in n1: #左闭右开
    img = im.open('/Users/ycy/Desktop/flower_frommysql/'+str(i)+'.jpg').convert("RGB") #L为灰度图，RGB为真彩色，CMYK为pre-press图像
    out = img.resize((64,64))
    arr = np.array(out)
    np.save('/Users/ycy/Desktop/array_train&test/arr/flower/test/'+str(i),arr)
    array_flower_test[j] = arr
    j = j + 1    
np.save('/Users/ycy/Desktop/array_train&test/image_flower_test',array_flower_test)

## image_bird_train
array_bird_train = np.zeros(((100,64,64,3)))
j = 0
for i in m2: #左闭右开
    img = im.open('/Users/ycy/Desktop/bird_frommysql/'+str(i)+'.jpg').convert("RGB") #L为灰度图，RGB为真彩色，CMYK为pre-press图像
    out = img.resize((64,64))
    arr = np.array(out)
    np.save('/Users/ycy/Desktop/array_train&test/arr/bird/train/'+str(i),arr)
    array_bird_train[j] = arr
    j = j + 1    
np.save('/Users/ycy/Desktop/array_train&test/image_bird_train',array_bird_train)

## image_bird_test
array_bird_test = np.zeros(((100,64,64,3)))
j = 0
for i in n2: #左闭右开
    img = im.open('/Users/ycy/Desktop/bird_frommysql/'+str(i)+'.jpg').convert("RGB") #L为灰度图，RGB为真彩色，CMYK为pre-press图像
    out = img.resize((64,64))
    arr = np.array(out)
    np.save('/Users/ycy/Desktop/array_train&test/arr/bird/test/'+str(i),arr)
    array_bird_test[j] = arr
    j = j + 1    
np.save('/Users/ycy/Desktop/array_train&test/image_bird_test',array_bird_test)

## image_insect_train
array_insect_train = np.zeros(((100,64,64,3)))
j = 0
for i in m3: #左闭右开
    img = im.open('/Users/ycy/Desktop/insect_frommysql/'+str(i)+'.jpg').convert("RGB") #L为灰度图，RGB为真彩色，CMYK为pre-press图像
    out = img.resize((64,64))
    arr = np.array(out)
    np.save('/Users/ycy/Desktop/array_train&test/arr/insect/train/'+str(i),arr)
    array_insect_train[j] = arr
    j = j + 1    
np.save('/Users/ycy/Desktop/array_train&test/image_insect_train',array_insect_train)

## image_insect_test
array_insect_test = np.zeros(((100,64,64,3)))
j = 0
for i in n3: #左闭右开
    img = im.open('/Users/ycy/Desktop/insect_frommysql/'+str(i)+'.jpg').convert("RGB") #L为灰度图，RGB为真彩色，CMYK为pre-press图像
    out = img.resize((64,64))
    arr = np.array(out)
    np.save('/Users/ycy/Desktop/array_train&test/arr/insect/test/'+str(i),arr)
    array_insect_test[j] = arr
    j = j + 1    
np.save('/Users/ycy/Desktop/array_train&test/image_insect_test',array_insect_test)

## image_fish_train
array_fish_train = np.zeros(((100,64,64,3)))
j = 0
for i in m4: #左闭右开
    img = im.open('/Users/ycy/Desktop/fish_frommysql/'+str(i)+'.jpg').convert("RGB") #L为灰度图，RGB为真彩色，CMYK为pre-press图像
    out = img.resize((64,64))
    arr = np.array(out)
    np.save('/Users/ycy/Desktop/array_train&test/arr/fish/train/'+str(i),arr)
    array_fish_train[j] = arr
    j = j + 1    
np.save('/Users/ycy/Desktop/array_train&test/image_fish_train',array_fish_train)

## image_fish_test
array_fish_test = np.zeros(((100,64,64,3)))
j = 0
for i in n4: #左闭右开
    img = im.open('/Users/ycy/Desktop/fish_frommysql/'+str(i)+'.jpg').convert("RGB") #L为灰度图，RGB为真彩色，CMYK为pre-press图像
    out = img.resize((64,64))
    arr = np.array(out)
    np.save('/Users/ycy/Desktop/array_train&test/arr/fish/test/'+str(i),arr)
    array_fish_test[j] = arr
    j = j + 1    
np.save('/Users/ycy/Desktop/array_train&test/image_fish_test',array_fish_test)


#--------------------------将train合成一个npy，test合成一个npy-----------------------------
## 生成image_train.npy
flower_train = np.load('/Users/ycy/Desktop/array_train&test/image_flower_train.npy')
bird_train = np.load('/Users/ycy/Desktop/array_train&test/image_bird_train.npy')
insect_train = np.load('/Users/ycy/Desktop/array_train&test/image_insect_train.npy')
fish_train = np.load('/Users/ycy/Desktop/array_train&test/image_fish_train.npy')

array_train = np.vstack((flower_train,bird_train,insect_train,fish_train)) #在列上合并 在行上合并np.hstack
#array.shape  #(400,64,64)
np.save('/Users/ycy/Desktop/image_train', array_train)


## 生成image_test.npy
flower_test = np.load('/Users/ycy/Desktop/array_train&test/image_flower_test.npy')
bird_test = np.load('/Users/ycy/Desktop/array_train&test/image_bird_test.npy')
insect_test = np.load('/Users/ycy/Desktop/array_train&test/image_insect_test.npy')
fish_test = np.load('/Users/ycy/Desktop/array_train&test/image_fish_test.npy')

array_test = np.vstack((flower_test,bird_test,insect_test,fish_test)) #在列上合并 在行上合并np.hstack
np.save('/Users/ycy/Desktop/image_test', array_test)





#显示图像
image_train = np.load('/Users/ycy/Desktop/image_train.npy')
image_train.shape
plt.imshow(image_train[377]) #ndarray以图片形式显示出来
image_train[377]

image_test = np.load('/Users/ycy/Desktop/image_test.npy')
image_test.shape
plt.imshow(image_test[157])
image_test[157]
