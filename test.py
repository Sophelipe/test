import os, sys
import caffe
import cv2
import numpy as np
from dataLayer import dataLoader
model = r'/caffe/models/bvlc_googlenet/bvlc_googlenet.caffemodel'
deploy = r'/caffe/models/bvlc_googlenet/deploy.prototxt'

caffe.set_mode_gpu()
caffe.set_device(0)
net = caffe.Net(deploy, model, caffe.TEST)
dataloader = dataLoader({'state':'test'})
img_shop, img_custom, label = dataloader.load_data()
print img_shop