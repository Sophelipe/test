#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2017-03-12 17:10:14
# @Author  : FelipeLi 

import os, sys
import caffe
import cv2
import numpy as np
from caffe import layers as L
from caffe import params as P
sys.path.insert(0,os.getcwd())

prototxt = './train.prototxt'
prototxt = './train_siamese.prototxt'
net = caffe.Net(prototxt, caffe.TRAIN)

# print "\nnet.inputs =", net.inputs
# print "\ndir(net.blobs) =", dir(net.blobs)
# print "\ndir(net.params) =", dir(net.params)
# print "\nconv shape = ", net.blobs['conv1'].data.shape

for i in range(10):
	net.forward()
	print net.blobs['img_shop'].data.shape
	# print net.blobs['img_cumstom'].data.shape
	print net.blobs['conv1'].data.shape
	# print net.blobs['pool5_spp'].data.shape
	# print net.blobs['ip1'].data.shape
	# print net.blobs['conv1_p'].data.shape
	img = net.blobs['img_shop'].data[0,...].transpose(1,2,0).astype('uint8')
	cv2.imshow("Image", img)
	# rsize = cv2.resize(img,(200, 300))
	# print rsize.shape
	cv2.waitKey(0)