#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2017-04-06 10:38:22
# @Author  : FelipeLi 

import os
import platform
imgDir = r'/dataset/DeepFashion/DeepFashion-Consumer-to-shop/'
if platform.system() == 'Windows':
	imgDir = r'D:/LHF/Clothing/DeepFashion/img/'

def check_params(params, required):
    """
    A utility function to check the parameters.
    """
    
    # required = ['batch_size',]
    for r in required:
        assert r in params.keys(), 'Params must include {}'.format(r)

def list_folder(path):
        """Folders only."""
        return [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
        # for entry in os.listdir(path):
        #     childpath = os.path.join(path, entry)
        #     if os.path.isdir(childpath):
        #         yield entry

def get_dir(dir, maxlevel):

	def get_dir1(dir, index, maxlevel, lis):
		if index+1 <= maxlevel:
			item = list_folder(dir)
			
			for d in item:
				get_dir1(os.path.join(dir, d), index+1, maxlevel, lis)

			lis[index].extend([x for x in item if x not in lis[index]]) #去除重复
	
	lis = [[] for i in range(maxlevel)]
	get_dir1(dir, 0, maxlevel, lis)
	return lis

class LableTool(object):
	def __init__(self, labelList):
		self._dict = []
		self._dict = dict(zip(labelList, xrange(len(labelList))))
		print self._dict
		self._list = range(len(labelList))
		for k,v in self._dict.items():
			self._list[v] = k

	def getLabelIndex(self, key):
		return self._dict[key]

	def getLabelName(self, index):
		return self._list[index]

if __name__ == '__main__':
	label = get_dir(imgDir, 2)
	labeltool = LableTool(label[1])