#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2017-04-10 11:19:58
# @Author  : FelipeLi 

import os
import platform
import itertools
import cv2

rootPath = r'/dataset/DeepFashion/DeepFashion-Consumer-to-shop/'
if platform.system() == 'Windows':
	rootPath = r'D:/LHF/Clothing/DeepFashion/'

bboxFile = os.path.join(rootPath, r'Anno/list_bbox_consumer2shop.txt')

with open(bboxFile) as f:
	num = int(f.readline())
	iter = itertools.islice(f, 1, None)
	count = 0
	for i in xrange(num):
		data = iter.next().split()
		imgName = os.path.join(rootPath, data[0])
		if os.path.exists(imgName):
			x1, y1, x2, y2 = int(data[2])-1, int(data[3])-1, int(data[4])-1, int(data[5])-1
			img = cv2.imread(imgName)
			cropImg = img[y1:y2, x1:x2]

			saveFile = os.path.join(rootPath, r'a',  data[0])
			savePath = os.path.dirname(saveFile) # 获取文件路径
			if not os.path.exists(savePath):
				os.makedirs(savePath)

			cv2.imwrite(saveFile, cropImg)