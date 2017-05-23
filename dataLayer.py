#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2017-03-12 16:45:57
# @Author  : FelipeLi 

import caffe
import os
import numpy as np
import cv2
import platform
import itertools
import random
from labelTool import get_dir,LableTool

state_dict = ['train', 'test']
class MydataLayer(caffe.Layer):
	
	def setup(self, bottom, top):
		print 'MydataLayer.setup begin'
		# self._batchSize = 2
		params = eval(self.param_str)

		# Check the paramameters for validity.
		check_params(params)
		# store input as class variables
		self._batchSize = params['batch_size']
		self._towlabel = params['towlabel']
		self._topnum = 3
		if self._towlabel != None and self._towlabel == True:
			self._topnum += 1
		self._meanValue = np.array([104,117,123], dtype = np.uint8)[:, np.newaxis, np.newaxis]

		if len(bottom) != 0:
			   raise Exception('must have no input')

		if len(top) != self._topnum :
			   raise Exception('must have exactly %s outputs' % self._topnum )

		
		self._dataloader = DataLoader({'state':state_dict[self.phase]})

		top[0].reshape(self._batchSize,3,224,224)
		top[1].reshape(self._batchSize,3,224,224)
		top[2].reshape(self._batchSize,1,1,1)
		if self._towlabel != None and self._towlabel == True:
			top[3].reshape(self._batchSize,2,1,1)

		# top[3].reshape(self._batchSize,1,1,1)
		# top[4].reshape(self._batchSize,1,1,1)
		print 'MydataLayer.setup end'

	def reshape(self,bottom,top):

		# print 'MydataLayer.reshape begin'
		pass
		# print 'MydataLayer.reshape end'

	def forward(self,bottom,top): 
		# print 'MydataLayer.forward begin'
		shoplist = []
		customlist = []
		simList = []
		towsimList = []
		labelShopList = []
		labelCustomList = []
		for i in range(self._batchSize):
			img_shop, img_custom, sim, label_shop, label_custom = self._dataloader.load_data(state_dict[self.phase])
			# img_shop -= self._meanValue
			# img_custom -= self._meanValue
			if self._towlabel != None and self._towlabel == True:
				towsimList.append([sim, 1-sim])

			shoplist.append(img_shop)
			customlist.append(img_custom)
			simList.append(sim)
			labelShopList.append(label_shop)
			labelCustomList.append(label_custom)

		shoplist = np.array(shoplist)
		customlist = np.array(customlist)
		simList = np.array(simList)
		labelShopList = np.array(labelShopList)
		labelCustomList = np.array(labelCustomList)

		simList = simList.reshape(self._batchSize,1,1,1)
		labelShopList = labelShopList.reshape(self._batchSize,1,1,1)
		labelCustomList = labelCustomList.reshape(self._batchSize,1,1,1)

		top[0].reshape(*shoplist.shape)
		top[1].reshape(*customlist.shape)

		# do your magic here... feed **one** batch to `top`
		top[0].data[...] = shoplist
		top[1].data[...] = customlist
		top[2].data[...] = simList

		if self._towlabel != None and self._towlabel == True:
			towsimList = np.array(towsimList)
			towsimList = towsimList.reshape(self._batchSize,2,1,1)
			top[3].data[...] = towsimList
		# top[3].data[...] = labelShopList
		# top[4].data[...] = labelCustomList

		#　print top[2].data[...].shape
		# print 'MydataLayer.forward end'
		
	def backward(self, top, propagate_down, bottom):
		# no back-prop for input layers
		# print 'MydataLayer.backward begin'
		pass
		# print 'MydataLayer.backward end'

rootPath = r'/dataset/DeepFashion/DeepFashion-Consumer-to-shop/'
model = r'/caffe/models/bvlc_googlenet/bvlc_googlenet.caffemodel'
deploy = r'/caffe/models/bvlc_googlenet/deploy.prototxt'	
if platform.system() == 'Windows':
	rootPath = r'D:/LHF/Clothing/DeepFashion/'
	model = r'../../models/bvlc_googlenet/bvlc_googlenet.caffemodel'
	deploy = r'../../models/bvlc_googlenet/deploy.prototxt'

evalFile = os.path.join(rootPath, r'Eval/list_eval_partition.txt.neg')

class MyfeatureLayer(caffe.Layer):
	def setup(self, bottom, top):
		print 'MyfeatureLayer.setup begin'
		
		if len(bottom) != 2:
			   raise Exception('must have 2 inputs')

		if len(top) != 2:
			   raise Exception('must have exactly 2 outputs')

		self._batchSize = len(bottom[0].data[...])

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

		self._net.blobs['data'].reshape(*net_input.shape)
		self._net.blobs['data'].data[...] = net_input
		self._net.forward()

		feature1 = self._net.blobs['pool5/7x7_s1'].data[:self._batchSize,...]
		feature2 = self._net.blobs['pool5/7x7_s1'].data[self._batchSize:,...]

		top[0].reshape(*feature1.shape)
		top[1].reshape(*feature2.shape)
		top[0].data[...] = feature1
		top[1].data[...] = feature2

	def backward(self,bottom,top):
		pass

class DataLoader(object):
 	"""docstring for ClassName"""
 	def __init__(self, param):
		print 'DataLoader.__init__'

		self._state = param['state']
		self._pre = {'file':'', 'img':''}
		label = get_dir(os.path.join(rootPath,'img'), 2)
		self._labelTool = LableTool(label[1])
		self.open_eval()

	def __del__(self):
		print 'DataLoader.__del__'
		if self._file:
			self._file.close()

	def open_eval(self):
		if hasattr(self,'_file') and self._file:
			print 're-open file:%s' % evalFile
			self._file.close()

		try:
			self._file = open(evalFile)
		except:  
			print "Failed to open file: %s" % evalFile
			exit()

		shuffle = True
		self._fileIter = itertools.islice(self._file, 2, None)
		if shuffle:
			# 打乱文件行顺序
			tmp = list(self._fileIter)

			for i in range(5):
				random.shuffle(tmp)

			self._fileIter = itertools.cycle(tmp)
			del tmp

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

		# similarity
		sim = 1
		labelPos = 5
		if len(data) >= labelPos:
			sim = int(data[labelPos-1])

		#print len(data),sim
		return img2, img1, sim

	def get_raw_data(self):
		line = self._fileIter.next()
		while not line.find(self._state)>0:
			line = self._fileIter.next()

		return line.split()


	def load_data(self, state = None):
		if state != None:
			self._state = state
		file1 = ""
		file2 = ""
		data = []
		while not (os.path.exists(file1) and os.path.exists(file2)):
			data = self.get_raw_data()
			file1 = os.path.join(rootPath,'a',data[0])
			file2 = os.path.join(rootPath,'a',data[1])

		file_path1 = os.path.basename(file1)
		file_path2 = os.path.basename(file2)

		label_shop = data[0].split('/')[2]
		label_shop = self._labelTool.getLabelIndex(label_shop)
		label_custom = data[1].split('/')[2]
		label_custom = self._labelTool.getLabelIndex(label_custom)
		img_shop, img_custom, sim = self.load_image2(file1, file2, data)
		return img_shop, img_custom, sim, label_shop, label_custom


def check_params(params):
    """
    A utility function to check the parameters for the data layers.
    """
    
    required = ['batch_size',]
    for r in required:
        assert r in params.keys(), 'Params must include {}'.format(r)