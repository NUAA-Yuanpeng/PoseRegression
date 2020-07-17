# -*- coding: utf-8 -*-
"""
Created on Thu Aug 10 15:21:19 2017

@author: Yuanpeng
"""

import os
import numpy as np
from matplotlib import pyplot as plt

def DrawQuaternions(gdt, pre, basepath, reload=False, ftype='', prefix=''):
    if reload:
        gdt = np.loadtxt(os.path.join(basepath, 'gdt_quaternions' + ftype + '.txt'))
        pre = np.loadtxt(os.path.join(basepath, 'pre_quaternions' + ftype + '.txt'))
    
    if gdt.shape[0] != pre.shape[0]:
        return
    
    nsize = 100
    if nsize > gdt.shape[0]:
        nsize = gdt.shape[0]
    
    idx = np.random.choice(gdt.shape[0], nsize, replace = False)
    gdt = gdt[idx, :]
    pre = pre[idx, :]
        
    gdt_quaternions_0 = gdt[:, 0]
    gdt_quaternions_1 = gdt[:, 1]
    gdt_quaternions_2 = gdt[:, 2]
    gdt_quaternions_3 = gdt[:, 3]
    
    pre_quaternions_0 = pre[:, 0]
    pre_quaternions_1 = pre[:, 1]
    pre_quaternions_2 = pre[:, 2]
    pre_quaternions_3 = pre[:, 3]

    error_0 = np.sqrt(np.sum(np.square(gdt_quaternions_0 - pre_quaternions_0))/gdt_quaternions_0.shape[0])
    error_1 = np.sqrt(np.sum(np.square(gdt_quaternions_1 - pre_quaternions_1))/gdt_quaternions_1.shape[0])
    error_2 = np.sqrt(np.sum(np.square(gdt_quaternions_2 - pre_quaternions_2))/gdt_quaternions_2.shape[0])
    error_3 = np.sqrt(np.sum(np.square(gdt_quaternions_3 - pre_quaternions_3))/gdt_quaternions_3.shape[0])
    error_mean = (error_0 + error_1 + error_2 + error_3)/4

    print('error_0: ', error_0)
    print('error_1: ', error_1)
    print('error_2: ', error_2)
    print('error_3: ', error_3)
    print('error_mean: ', error_mean)

    x_axis = list(range(0, nsize))
    figure_dir = os.path.join(basepath, 'Figures')
    if not os.path.exists(figure_dir):
        os.mkdir(figure_dir)
		
    error_file_path = os.path.join(figure_dir, 'error' + ftype + '.txt')
    error_file = open(error_file_path, 'w')
    error_file.write(str('error_0: ') + str(error_0) + '\n')
    error_file.write(str('error_1: ') + str(error_1) + '\n')
    error_file.write(str('error_2: ') + str(error_2) + '\n')
    error_file.write(str('error_3: ') + str(error_3) + '\n')
    error_file.write(str('error_mean: ') + str(error_mean) + '\n')
    error_file.close()
	
    prefix_path = os.path.join(figure_dir, prefix)
    
    fg_0=plt.figure('error_quaternions_0')      
    plt.plot(x_axis, gdt_quaternions_0.tolist(), 'r', pre_quaternions_0.tolist(), 'b', label='quaternions_0')
    plt.savefig(prefix_path + '_quaternions_0' + ftype + '.png', dpi = 120)
    fg_1=plt.figure('error_quaternions_1')      
    plt.plot(x_axis, gdt_quaternions_1.tolist(), 'r', pre_quaternions_1.tolist(), 'b', label='quaternions_1')
    plt.savefig(prefix_path + '_quaternions_1' + ftype + '.png', dpi = 120)
    fg_2=plt.figure('error_quaternions_2')      
    plt.plot(x_axis, gdt_quaternions_2.tolist(), 'r', pre_quaternions_2.tolist(), 'b', label='quaternions_2')
    plt.savefig(prefix_path + '_quaternions_2' + ftype + '.png', dpi = 120)
    fg_3=plt.figure('error_quaternions_3')      
    plt.plot(x_axis, gdt_quaternions_3.tolist(), 'r', pre_quaternions_3.tolist(), 'b', label='quaternions_3')
    plt.savefig(prefix_path + '_quaternions_3' + ftype + '.png', dpi = 120)
    #plt.show()
    
#DrawQuaternions(0, 0, reload=True)