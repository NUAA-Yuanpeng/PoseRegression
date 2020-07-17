# -*- coding: utf-8 -*-
"""
Created on Sat Aug 26 14:01:20 2017

# 20170925: add dividing depth images 

@author: Yuanpeng
"""

import shutil
import os
import numpy as np
from PIL import Image

base_dir = './dataset/Images'
obj_name = '/Cat'
binary_dir = '/Binary'
depth_dir = '/Depth'                # 2017_09_22
binary_base = base_dir + obj_name + binary_dir
depth_base = base_dir + obj_name + depth_dir
train_dir_bin = binary_base + '/train'
test_dir_bin = binary_base + '/test'
train_dir_dep = depth_base + '/train'
test_dir_dep = depth_base + '/test'

latter_bin = '-imageDepth-0000_bin.png'
latter_dep = '-imageDepth-0000.png'

print('Object:' + obj_name)
print('Loading poses...')
poses = np.loadtxt(base_dir + obj_name + '/poses.txt')
total_num = poses.shape[0]
num_array = np.random.choice(total_num, total_num, replace = False)
test_num = int(total_num * 0.1)
test_idx = num_array[:test_num]
train_idx = num_array[test_num:]

pose_train = poses[train_idx]
pose_test = poses[test_idx]

print('Generating filenames...')
files_bin_train = []
files_bin_test = []
files_dep_train = []
files_dep_test = []
for i in test_idx:
    filename_bin_test = binary_base + '/' + str(i) + latter_bin
    filename_dep_test = depth_base + '/' + str(i) + latter_dep
    files_bin_test.append(filename_bin_test)
    files_dep_test.append(filename_dep_test)
for i in train_idx:
    filename_bin_train = binary_base + '/' + str(i) + latter_bin
    filename_dep_train = depth_base + '/' + str(i) + latter_dep
    files_bin_train.append(filename_bin_train)
    files_dep_train.append(filename_dep_train)

print('Loading images...')
imgs_bin_train = []
imgs_bin_test = []
imgs_dep_train = []
imgs_dep_test = [] 
for i in files_bin_train:
    img = Image.open(i)
    grey = img.convert('L')
    img = np.array(grey.resize((48, 48)))
    aver = float(img[23, 23])
    imgs_bin_train.append(img.astype('float32')/aver)
for i in files_bin_test:
    img = Image.open(i)
    grey = img.convert('L')
    img = np.array(grey.resize((48, 48)))
    aver = float(img[23, 23])
    imgs_bin_test.append(img.astype('float32')/aver)
for i in files_dep_train:
    img = Image.open(i)
    img = np.array(img.resize((48, 48)))
    aver = float(img[23, 23])
    if aver < 0.1:
        print("error")
    imgs_dep_train.append(img.astype('float32')/aver)
for i in files_dep_test:
    img = Image.open(i)
    img = np.array(img.resize((48, 48)))
    aver = float(img[23, 23])
    imgs_dep_test.append(img.astype('float32')/aver)
imgs_bin_train = np.array(imgs_bin_train)
imgs_bin_test = np.array(imgs_bin_test)
imgs_dep_train = np.array(imgs_dep_train)
imgs_dep_test = np.array(imgs_dep_test)
#imgs_bin_train.resize(imgs_bin_train.shape[0], 48*48)
#imgs_bin_test.resize(imgs_bin_test.shape[0], 48*48)
#imgs_dep_train.resize(imgs_dep_train.shape[0], 48*48)
#imgs_dep_test.resize(imgs_dep_test.shape[0], 48*48)
    
print('Saving data...')
np.save(base_dir + obj_name + '/imgs_train.npy', imgs_bin_train)
np.save(base_dir + obj_name + '/poses_train.npy', pose_train)
np.save(base_dir + obj_name + '/depths_train.npy', imgs_dep_train)
np.save(base_dir + obj_name + '/imgs_test.npy', imgs_bin_test)
np.save(base_dir + obj_name + '/poses_test.npy', pose_test)
np.save(base_dir + obj_name + '/depths_test.npy', imgs_dep_test)