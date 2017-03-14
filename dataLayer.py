#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2017-03-12 16:45:57
# @Author  : FelipeLi 

import caffe
import os
import numpy as np
import cv2
from itertools import islice

class MydataLayer(caffe.Layer):
	def setup(self, bottom, top):
		print 'MydataLayer.setup begin'
		self._batchSize = 1
		if len(bottom) != 0:
			   raise Exception('must have no input')

		if len(top) != 3:
			   raise Exception('must have exactly 3 outputs')

		state_dict = ['train', 'test']
		self._dataLoader = dataLoader({'state':state_dict[self.phase]})
		top[0].reshape(self._batchSize,3,300,200)
		top[1].reshape(self._batchSize,3,300,200)
		top[2].reshape(self._batchSize,1)
		print 'MydataLayer.setup end'

	def reshape(self,bottom,top):

		print 'MydataLayer.reshape begin'
		pass
		print 'MydataLayer.reshape end'

	def forward(self,bottom,top): 
		print 'MydataLayer.forward begin'
		img_shop, img_cumstion, label = self._dataLoader.load_data()
		top[0].reshape(self._batchSize,*img_shop.shape)
		top[1].reshape(self._batchSize,*img_cumstion.shape)

		# do your magic here... feed **one** batch to `top`
		top[0].data[0,...] = img_shop
		top[1].data[0,...] = img_cumstion
		top[2].data[0,...] = label
		print 'MydataLayer.forward end'
		
	def backward(self, top, propagate_down, bottom):
		# no back-prop for input layers
		print 'MydataLayer.backward begin'
		pass
		print 'MydataLayer.backward end'

rootPath = r'D:\LHF\Clothing\DeepFashion'
evalFile = os.path.join(rootPath, r'Eval\list_eval_partition.txt')

class dataLoader(object):
 	"""docstring for ClassName"""
 	def __init__(self, param):
		print 'dataLoader.__init__'

		self._state = param['state']
		self._pre = {'file':'', 'img':''}
		try:
			self._file = open(evalFile)
			self._fileIter = islice(self._file, 2, None)
		except:  
			print "Failed to open file: %s" % evalFile
			exit()

	def __del__(self):
		print 'dataLoader.__del__'
		if self._file:
			self._file.close()

	def load_data(self):
		line = self._fileIter.next()
		while not line.find(self._state)>0:
			line = self._fileIter.next()

		data = line.split()

		# custom image
		file1 = os.path.join(rootPath,data[0])
		# img1 = cv2.imread(file1).transpose(2,0,1)
		img1 = cv2.imread(file1)
		img1 = cv2.resize(img1,(200, 300)).transpose(2,0,1)

		# shop image
		if data[1] == self._pre['file']:
			img2 = self._pre['img']

		else:
			file2 = os.path.join(rootPath,data[1])

			# img2 = cv2.imread(file2).transpose(2,0,1)
			img2 = cv2.imread(file2)
			img2 = cv2.resize(img2,(200, 300)).transpose(2,0,1)
			self._pre['file'] = data[1]
			self._pre['img'] = img2

		return img2, img1, 1