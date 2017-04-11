#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2017-04-11 09:17:30
# @Author  : FelipeLi 

import os
import platform
import itertools
import cv2
import random

rootPath = r'/dataset/DeepFashion/DeepFashion-Consumer-to-shop/'
if platform.system() == 'Windows':
	rootPath = r'D:/LHF/Clothing/DeepFashion/'

evalFile = os.path.join(rootPath, r'Eval/list_eval_partition.txt')
negative = 3
with open(evalFile) as f:
	num = int(f.readline())

	fileIter = itertools.islice(f, 1, None)
	tmp = list(fileIter)
	fileIter = itertools.islice(tmp[:], None, None) # 原始顺序文件
	for i in range(5):
		random.shuffle(tmp)

	suffleIter = itertools.cycle(tmp) # 乱序文件

	del tmp

	try:
		filetowrite = os.path.join(rootPath,evalFile + ".neg")

		if os.path.isfile(filetowrite):
			os.remove(filetowrite)

		f_neg = open(filetowrite, 'w+')
	except:
		print "Failed to open file: %s" % f
		exit()

	split = ("\t"*2)
	for i in range(num):
		line = fileIter.next()
		data = line.split()
		data.append("1\n") # 打上正样本标签
		f_neg.writelines(split.join([x for x in data]))
		shopStr = data[1]
		shop = shopStr.split('/')
		# state = data[3]
		
		count = 0
		neglist = []
		loopcount = 0;
		if i % 200 == 0:
			print i
		while count < negative: 
			negData = suffleIter.next().split()
			negCustomStr = negData[0]
			negShopStr = negData[1]
			negShop = negShopStr.split('/')

			# negState = negData[3]
			if (data[3] == negData[3]) \
			and (shop[1:3] == negShop[1:3]) \
			and (shop[3] != negShop[3]) \
			and not (negCustomStr in neglist):

				count  += 1
				neglist.append(negCustomStr)
				sample = [negCustomStr, shopStr, "no_pairs", data[3] , "0\n"]
				f_neg.writelines(split.join([x for x in sample]))
			elif (shop[1:3] != negShop[1:3]):
				loopcount += 1
				if loopcount > 2000:
					count = negative
					print i,'do not success: %s' % line
					continue
	f_neg.close()

