# import os, sys
# import caffe
# import cv2
# import numpy as np
# from dataLayer import DataLoader
# import platform
# from itertools import islice

# model = r'../../models/bvlc_googlenet/bvlc_googlenet.caffemodel'
# deploy = r'../../models/bvlc_googlenet/deploy.prototxt'

# # caffe.set_mode_gpu()
# # caffe.set_device(0)
# # net = caffe.Net(deploy, model, caffe.TEST)
# dataloader = DataLoader({'state':'test'})
# batch_size = 10;
# shoplist = []
# customlist = []
# for i in range(batch_size):
# 	img_shop, img_custom, label = dataloader.load_data()
# 	customlist.append(img_custom)
# 	print customlist

# customlist = np.array(customlist)

# # net.blobs['data'].reshape(*customlist.shape)
# # net.blobs['data'].data[...] = customlist
# # print len(customlist)
# # net.forward()

# # print net.blobs['pool5/7x7_s1'].data[...].shape
# # print net.blobs['pool5/7x7_s1'].data[0,...]

import random,os
import itertools
rootPath = r'D:/LHF/Clothing/DeepFashion/'
evalFile = os.path.join(rootPath, r'Eval/list_eval_partition.txt')

_file = open(evalFile)

_fileIter = itertools.islice(_file, 2, None)



a = list(_fileIter)

random.shuffle(a)
c_fileIter = itertools.cycle(a)
del a
for i in range(10):
	print c_fileIter.next()