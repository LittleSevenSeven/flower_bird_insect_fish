#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 21:32:35 2017

@author: ycy
"""

import pymysql
import sys
import os

    
def iterbrowse(path):
    filelist = []
    for root, dirs, files in os.walk(path): # os.walk()遍历文件夹下所有的文件
        for filename in files:
            temp = os.path.join(root, filename) # os.path.join：将多个路径组合后返回
            #print(temp)
            if temp[-1] != 'e': # 删除.DS_store隐藏文件
                filelist.append(temp)
                
                fin = open(temp,'rb') #读取文件的这一步写在if语句里
                img = fin.read()  
                fin.close()
    
                conn = pymysql.connect(host="localhost", port=3306, user="root", \
                                       passwd="ycy", db="pictures") #连接到数据库  
                cursor = conn.cursor()
                cursor.execute('use pictures')
                cursor.execute("insert into hncy4(class, image, website) values (%s, %s, %s)",\
                               ('fish',pymysql.Binary(img),root[24:]))
                conn.commit()   #提交数据  
                cursor.close()  
                conn.close()             
            else:
                pass
    return filelist


if __name__ == '__main__':
    filelist = iterbrowse('/Users/ycy/Desktop/fish')
    print(filelist)
    print(len(filelist))
