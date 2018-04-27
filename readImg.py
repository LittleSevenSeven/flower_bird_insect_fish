#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 19 23:01:42 2017

@author: ycy
"""

import pymysql 
import sys


conn = pymysql.connect(host='localhost', port=3306, user='root', passwd='ycy', db='pictures')
cur = conn.cursor()
cur.execute('use pictures')
cur.execute("select * from hncy4 where class='fish'")

data = cur.fetchall()
cur.close()
conn.close()

id = [row[0] for row in data] #row为数据库里每一行数据
image = [hang[2] for hang in data]


for i in id:
    f = open('/Users/ycy/Desktop/fish_frommysql/'+str(i-600)+'.jpg',"wb") #创建文件
    f.write(image[i-601]) #对fish来说，id是601～800
    f.close()


### 本code用来存储的flower为1～200，bird为1～200，insect为1～200，fish为1～200