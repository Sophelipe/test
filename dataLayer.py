#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2017-03-12 16:45:57
# @Author  : FelipeLi 

import caffe
import os
import numpy as np
import cv2
import platform
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
		self._dataloader = DataLoader({'state':state_dict[self.phase]})
		top[0].reshape(self._batchSize,3,300,200)
		top[1].reshape(self._batchSize,3,300,200)
		top[2].reshape(self._batchSize,1,1,1)
		print 'MydataLayer.setup end'

	def reshape(self,bottom,top):

		# print 'MydataLayer.reshape begin'
		pass
		# print 'MydataLayer.reshape end'

	def forward(self,bottom,top): 
		# print 'MydataLayer.forward begin'
		shoplist = []
		customlist = []
		labellist = []
		for i in range(self._batchSize):
			img_shop, img_custom, label = self._dataloader.load_data()
			shoplist.append(img_shop)
			customlist.append(img_custom)
			labellist.append(1)

		shoplist = np.array(shoplist)
		customlist = np.array(customlist)
		labellist = np.array(labellist)
		labellist = labellist.reshape(self._batchSize,1,1,1)

		img_shop, img_cumstion, label = self._dataloader.load_data()
		top[0].reshape(*shoplist.shape)
		top[1].reshape(*customlist.shape)

		# do your magic here... feed **one** batch to `top`
		top[0].data[...] = shoplist
		top[1].data[...] = customlist
		top[2].data[...] = labellist

		#　print top[2].data[...].shape
		# print 'MydataLayer.forward end'
		
	def backward(self, top, propagate_down, bottom):
		# no back-prop for input layers
		# print 'MydataLayer.backward begin'
		pass
		# print 'MydataLayer.backward end'

rootPath = r'/dataset/DeepFashion/DeepFashion-Consumer-to-shop/'
if platform.system() == 'Windows':
	rootPath = r'D:/LHF/Clothing/DeepFashion/'

evalFile = os.path.join(rootPath, r'Eval/list_eval_partition.txt')

class MyfeatureLayer(caffe.Layer):
	def setup(self, bottom, top):
		print 'MyfeatureLayer.setup begin'
		
		if len(bottom) != 2:
			   raise Exception('must have 2 inputs')

		if len(top) != 2:
			   raise Exception('must have exactly 2 outputs')

		self._batchSize = len(bottom[0].data[...])

		model = r'../../models/bvlc_googlenet/bvlc_googlenet.caffemodel'
		deploy = r'../../models/bvlc_googlenet/deploy.prototxt'

		self._net = caffe.Net(deploy, model, caffe.TEST)
		top[0].reshape(self._batchSize,1024,1,1)
		top[1].reshape(self._batchSize,1024,1,1)
		print 'MyfeatureLayer.setup end'

	def reshape(self,bottom,top):
		pass

	def forward(self,bottom,top): 
		net_input = []
		net_input.extend(bottom[0].data[...])
		net_input.extend(bottom[1].data[...])
		net_input = np.array(net_input)
		# print net_input.shape
		self._net.blobs['data'].reshape(*net_input.shape)
		self._net.blobs['data'].data[...] = net_input
		self._net.forward()

		feature1 = self._net.blobs['pool5/7x7_s1'].data[:self._batchSize,...]
		feature2 = self._net.blobs['pool5/7x7_s1'].data[self._batchSize:,...]

		top[0].reshape(*feature1.shape)
		top[0].reshape(*feature2.shape)
		
	def backward(self,bottom,top):
		pass

class DataLoader(object):
 	"""docstring for ClassName"""
 	def __init__(self, param):
		print 'DataLoader.__init__'

		self._state = param['state']
		self._pre = {'file':'', 'img':''}
		try:
			self._file = open(evalFile)
			self._fileIter = islice(self._file, 2, None)
		except:  
			print "Failed to open file: %s" % evalFile
			exit()

	def __del__(self):
		print 'DataLoader.__del__'
		if self._file:
			self._file.close()

	def load_image2(self, file1, file2, data):
		
		# custom image
		# img1 = cv2.imread(file1).transpose(2,0,1)
		img1 = cv2.imread(file1)
		img1 = cv2.resize(img1,(224, 224)).transpose(2,0,1)

		# shop image
		if data[1] == self._pre['file']:
			img2 = self._pre['img']
		else:
			if not os.path.isfile(file2):
				return self.load_data()
			# img2 = cv2.imread(file2).transpose(2,0,1)
			img2 = cv2.imread(file2)
			img2 = cv2.resize(img2,(224, 224)).transpose(2,0,1)
			self._pre['file'] = data[1]
			self._pre['img'] = img2

		return img2, img1, 1

	def get_raw_data(self):
		line = self._fileIter.next()
		while not line.find(self._state)>0:
			line = self._fileIter.next()

		return line.split()

	def load_data(self):
		file1 = ""
		file2 = ""
		while not (os.path.exists(file1) and os.path.exists(file2)):
			data = self.get_raw_data()
			file1 = os.path.join(rootPath,data[0])
			file2 = os.path.join(rootPath,data[1])
			file_path1 = os.path.basename(file1)
			file_path2 = os.path.basename(file2)

		return self.load_image2(file1, file2, data)