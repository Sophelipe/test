#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2017-03-12 16:45:57
# @Author  : FelipeLi 

import sys
sys.path.append(r'/caffe/python/')
import caffe
import os
import numpy as np
import cv2
import platform
import itertools
import random
import h5py as h5
from labelTool import get_dir,LableTool

state_dict = ['train', 'test']
rootPath = r'/dataset/DeepFashion/DeepFashion-Consumer-to-shop/'
model = r'/caffe/models/bvlc_googlenet/bvlc_googlenet.caffemodel'
deploy = r'/caffe/models/bvlc_googlenet/deploy.prototxt'
meanfile = r'/caffe/python/caffe/imagenet/ilsvrc_2012_mean.npy'


if platform.system() == 'Windows':
	rootPath = r'D:/LHF/Clothing/DeepFashion/'
	model = r'../../models/bvlc_googlenet/bvlc_googlenet.caffemodel'
	deploy = r'../../models/bvlc_googlenet/deploy.prototxt'

evalFile = os.path.join(rootPath, r'Eval/list_eval_partition.txt.neg')
class MydataLayer(caffe.Layer):
	
	def setup(self, bottom, top):
		print 'MydataLayer.setup begin'
		# self._batchSize = 2
		params = eval(self.param_str)

		# Check the paramameters for validity.
		check_params(params)
		# store input as class variables
		self._batchSize = params['batch_size']
		self._towlabel = params.get('towlabel', False)
		self._topnum = 3
		if self._towlabel == True:
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

		self._transformer = getTransformer(top[0].shape)

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
			# img_shop = self._transformer.preprocess('data', img_shop)
			# img_custom = self._transformer.preprocess('data', img_custom)
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

def getTransformer(shape):
	transformer = caffe.io.Transformer({'data': shape})
	transformer.set_transpose('data', (2,0,1))
	transformer.set_mean('data', np.load(meanfile).mean(1).mean(1))
	transformer.set_raw_scale('data', 255)
	transformer.set_channel_swap('data', (2,1,0))

	return transformer


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
		tmp = list(self._fileIter)
		train = []
		test = []
		val = []

		for line in tmp:
			data = line.split()
			file1 = os.path.join(rootPath,'a',data[0])
			file2 = os.path.join(rootPath,'a',data[1])
			if (os.path.exists(file1) and os.path.exists(file2)):
				if line.find('test')>0:
					test.append(line)
				elif line.find('train')>0:
					train.append(line)
				else:
					val.append(line)
		
		# sizeLevevl = 10000
		# test = train[4*sizeLevevl:5*sizeLevevl]
		# train = train[:4*sizeLevevl]

		if shuffle:
			# 打乱文件行顺序
			for i in range(5):
				random.shuffle(train)
				random.shuffle(test)
				random.shuffle(val)

		self.test_fileIter = itertools.cycle(test)
		self.train_fileIter = itertools.cycle(train)
		self.val_fileIter = itertools.cycle(val)
		self._fileIter = self.test_fileIter

		print "size = %s, test = %s, val = %s" % (len(train), len(test), len(val))
		del tmp, test, train, val

	def load_image(self, file):
		img = cv2.imread(file)
		img = cv2.resize(img,(224, 224)).transpose(2,0,1)

		return img

	def load_image2(self, file1, file2, data):
		
		# custom image
		img1 = self.load_image(file1)
		# img1 = caffe.io.load_image(file1)

		# shop image
		if data[1] == self._pre['file']:
			img2 = self._pre['img']
		else:
			if not os.path.isfile(file2):
				return self.load_data()
			# img2 = cv2.imread(file2).transpose(2,0,1)
			img2 = self.load_image(file2)
			# img2 = caffe.io.load_image(file2)
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
		if self._state == 'test':
			self._fileIter = self.test_fileIter
		elif self._state == 'train':
			self._fileIter = self.train_fileIter
		else:
			self._fileIter = self.val_fileIter

		line = self._fileIter.next()
		return line.split()


	def load_data(self, state = None):
		if state != None:
			self._state = state
			
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

	def getFileIter(self, state):
		it = None
		if state == 'test':
			it = self.test_fileIter
		if state == 'train':
			it = self.train_fileIter
		else:
			it = self.val_fileIter

		return it

class DataProcessLayer(caffe.Layer):
	
	def setup(self, bottom, top):
		print 'DataProcessLayer.setup begin'
		params = eval(self.param_str)

		# Check the paramameters for validity.
		# check_params(params, ['datasize'])
		self._isCrossLabel = params.get('crosslabel', False)

		self._batchSize = len(bottom[0].data[...])
		self._batchSize = self._batchSize*2
		self._topnum = 3
		
		if self._isCrossLabel:
			self._topnum += 1

		if len(bottom) != 3:
			   raise Exception('must have 3 input')

		if len(top) != self._topnum :
			   raise Exception('must have exactly %s outputs' % self._topnum )

		imageNum = 2
		for i in xrange(imageNum):
			top[i].reshape(self._batchSize, 3, 224, 224)

		top[imageNum].reshape(self._batchSize,1,1,1)

		if self._isCrossLabel:
			top[self._topnum-1].reshape(self._batchSize,2,1,1)

		print 'DataProcessLayer.setup end'

	def reshape(self,bottom,top):

		# print 'MydataLayer.reshape begin'
		pass
		# print 'MydataLayer.reshape end'

	def forward(self,bottom,top): 
		# print 'MydataLayer.forward begin'
		simList = [1 for x in xrange(self._batchSize/2)]
		simList.extend([0 for x in xrange(self._batchSize/2)])
		simList = np.array(simList, dtype = np.int8)
		simList = simList.reshape(self._batchSize,1,1,1)
		

		top[0].data[:self._batchSize/2, ...] = bottom[0].data[...]
		top[0].data[self._batchSize/2:, ...] = np.copy(bottom[0].data[...])

		top[1].data[:self._batchSize/2, ...] = bottom[1].data[...]
		top[1].data[self._batchSize/2:, ...] = bottom[2].data[...]

		top[2].data[...] = simList
		
		if self._isCrossLabel:
			crosslabelList = np.concatenate((1-simList, simList) ,1)
			top[3].data[...] = crosslabelList

		# shuffle
		index = range(self._batchSize)
		random.shuffle(index)
		for i in xrange(self._topnum):
			top[i].data[...] = top[i].data[index, ...]
		
	def backward(self, top, propagate_down, bottom):
		# no back-prop for input layers
		# print 'MydataLayer.backward begin'
		pass
		# print 'MydataLayer.backward end'

class DataRecordLayer(caffe.Layer):
	
	def setup(self, bottom, top):
		self._file = open(r'/caffe/record.txt', 'w')

	def reshape(self,bottom,top):
		pass

	def forward(self,bottom,top):
		self._file.write(np.array2string(bottom[0].data[...]) + '\n')

	def backward(self, top, propagate_down, bottom):
		pass

class NegativeCacheLayer(caffe.Layer):
	
	def setup(self, bottom, top):
		print 'NegativeCacheLayer.setup begin'
		params = eval(self.param_str)

		# Check the paramameters for validity.
		check_params(params, ['batch_size','data_size',])

		self._datasize = int(params['data_size'])
		self._type = params.get('type', 'custom')
		self._processNum = 0
		self._index = int(self._processNum/self._datasize)%10
		self._batchSize = int(params['batch_size'])

		self._imgList = None
		self._trainList = self.getNewNegative('train', 0)
		self._testList = self.getNewNegative('test', 0)
		self._imgList = self._trainList
		self._bottom = 1
		if len(bottom) != self._bottom:
			   raise Exception('must have %s input' % self._bottom )

		if len(top) != 1 :
			   raise Exception('must have exactly 1 outputs')

		top[0].reshape(self._batchSize, 3, 224, 224)

		print 'NegativeCacheLayer.setup end'

	def reshape(self,bottom,top):

		# print 'MydataLayer.reshape begin'
		pass
		# print 'MydataLayer.reshape end'

	def forward(self,bottom,top): 
		if state_dict[self.phase] == 'train':
			curIndex = int(self._processNum/self._datasize)%10
			if self._index != curIndex:
				self._index = curIndex
				self._trainList = self.getNewNegative('train', self._index)
				
			self._imgList = self._trainList
		else:
			self._imgList = self._testList

		index = random.sample(xrange(1000), self._batchSize)
		negative = self._imgList[index]
		negative = np.array(negative, dtype = np.float32)

		# do your magic here... feed **one** batch to `top`
		top[0].data[...] = negative

		if state_dict[self.phase] == 'train':
			self._processNum += self._batchSize
			if self._bottom  > 0:
				self._imgList[index] = np.copy(bottom[0].data[...])
		
	def backward(self, top, propagate_down, bottom):
		# no back-prop for input layers
		# print 'MydataLayer.backward begin'
		pass
		# print 'MydataLayer.backward end'

	def getNewNegative(self, state = 'train', index = 0):
		imgList = self._imgList
		imghdf = r'/caffe/examples/tes/data/negative/%s_custom_%s.h5' % (state, index)
		print 'get new negative: %s' % imghdf
		with h5.File(imghdf,'r') as h:
			imgList = h['data'][:]

		return imgList

def check_params(params, required = None):
    """
    A utility function to check the parameters for the data layers.
    """
    
    required = required or ['batch_size',]
    for r in required:
        assert r in params.keys(), 'Params must include {}'.format(r)