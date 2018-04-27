#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 10 21:32:07 2017

@author: ycy
"""

import urllib.request
import re


#第一步：获取要爬取的母网页的内容
url = "https://www.zhihu.com/question/23360831"
webPage=urllib.request.urlopen(url)
data = webPage.read()
data = data.decode('utf-8') # utf-8 or gbk

#第二步：对母网页内容处理，提取出里面的图片链接
k = re.split(r'\s+',data) 
s = [] 
sp = [] 
for i in k : 
    if (re.match(r'src',i) or re.match(r'href',i)): 
        if (not re.match(r'href="#"',i)): 
            if (re.match(r'.*?png"',i) or re.match(r'.*?jpg"',i)):
                    s.append(i) 

for it in s :
    if (re.match(r'.*?jpg"',it)): #找到jpg格式的图片，也可以修改为.png来找png格式的图片
        sp.append(it) 

#第三步：获取这些图片链接的内容，并保存成本地图片
cnt = 0
cou = 1
for it in sp: 
    m = re.search(r'src="(.*?)"',it)
    iturl = m.group(1)
    print(iturl)
    if (iturl[0]=='/'):
        continue;
    web = urllib.request.urlopen(iturl)
    itdata = web.read()
    if (cou<30):
        f = open('/Users/ycy/Desktop/fish/1/'+str(cou)+'.jpg',"wb") #创建文件
        cou = cou+1
        f.write(itdata)
        f.close()
        print(it)
        cnt = cnt+1