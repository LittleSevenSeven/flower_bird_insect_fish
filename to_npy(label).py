#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 20 00:00:11 2017

@author: ycy
"""

import numpy as np
#import PIL
#import PIL.Image as im
#import matplotlib.pyplot as plt
#import random

#----------------------------label.npy------------------------------
## label_train
a = np.zeros(100,int)
b = np.ones(100,int)
c = np.ones(100,int)*2
d = np.ones(100,int)*3

e = np.hstack((a,b,c,d)) #在行上合并 在列上合并np.vstack
e.shape
#print(e)

np.save('/Users/ycy/Desktop/label_train',e) 
#label = np.load('/Users/ycy/Documents/课件/常用软件箱及应用/谭军大作业/CNN/label.npy')



## label_test
np.save('/Users/ycy/Desktop/label_test',e) 