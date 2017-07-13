import os, sys
os.system(r'source /etc/profile')

# import caffe
# import cv2
# import numpy as np
# from dataLayer import DataLoader
# import platform
# from itertools import islice


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

# def initilize():
# 	model = r'../../models/bvlc_googlenet/bvlc_googlenet.caffemodel'
# 	deploy = r'../../models/bvlc_googlenet/deploy.prototxt'

# 	caffe.set_mode_gpu()
# 	caffe.set_device(0)
# 	net = caffe.Net(deploy, model, caffe.TEST)

# 	return net

# def getNetDetails(image, net):
# 	pass

import h5py as h5
import random
import numpy as np
imghdf = r'/caffe/examples/tes/data/custom_image_1000.h5'
data = []
id = []
with h5.File(imghdf,'r') as h:
	data = h['data'][:]
	id = h['id'][:]

	# imgList = dict(zip(id, data))
data = np.zeros((3,224,224), dtype = np.uint8)
index = range(0,9)
random.shuffle(index)
data[index[0:5]] = random.sample(data, 5)
print data[index]


