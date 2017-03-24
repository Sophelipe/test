import os, sys
import caffe
import cv2
import numpy as np
from dataLayer import DataLoader
model = r'../../models/bvlc_googlenet/bvlc_googlenet.caffemodel'
deploy = r'../../models/bvlc_googlenet/deploy.prototxt'

caffe.set_mode_gpu()
caffe.set_device(0)
net = caffe.Net(deploy, model, caffe.TEST)
dataloader = DataLoader({'state':'test'})
batch_size = 10;
shoplist = []
customlist = []
for i in range(batch_size):
	img_shop, img_custom, label = dataloader.load_data()
	customlist.append(img_custom)

customlist = np.array(customlist)

net.blobs['data'].reshape(*customlist.shape)
net.blobs['data'].data[...] = customlist
print len(customlist)
net.forward()

print net.blobs['pool5/7x7_s1'].data[...].shape
print net.blobs['pool5/7x7_s1'].data[0,...]