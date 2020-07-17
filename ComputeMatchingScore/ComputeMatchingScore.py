"""
Created on Sat Oct 27 20:22:10 2018

@author: Yuanpeng
"""

import numpy as np
import os
from tqdm import tqdm
from pyquaternion import Quaternion
from sixd import load_sixd

obj_dict = {'ape':'Ape2',               'benchviseblue':'Benchviseblue2',   'bowl':'Bowl2',             'camera':'Camera2', 
            'can':'Can2',               'cat':'Cat2',                       'cup':'Cup2',               'drill':'Drill2', 
            'duck':'Duck2',             'eggbox':'Eggbox2',                 'glue':'Glue2',             'holepuncher':'Holepuncher2', 
            'iron':'Iron2',             'lamp':'Lamp2',                     'phone':'Phone2'}
# ,           'elephent':'Elephent',
#            'bulldog':'My_Bulldog',     'mycat':'My_Cat',                     'dog':'My_Dog',             'leopard':'My_Leopard',     
#            'pig':'My_Pig'
#            }

basedir = '../Images'
sequence = 0

for key in obj_dict:
    sequence = sequence + 1
    obj_name = obj_dict[key]
    print('\nNow processing ', obj_name)
    print('sequence:', sequence)
    basepath = os.path.join(basedir, obj_name)
    if not os.path.exists(basepath):
        print('Missing folder:', obj_name)
        continue
    try:
        gdt = np.loadtxt(os.path.join(basepath, 'gdt_quaternions_pair.txt'))
        pre_depth = np.loadtxt(os.path.join(basepath, 'pre_quaternions_depth.txt'))
        pre_triplet = np.loadtxt(os.path.join(basepath, 'pre_quaternions_triplet.txt'))
    except:
        print('Missing files:', obj_name)
        continue

    bench = load_sixd("E:\projects\#2\datasets\hinterstoisser", nr_frames=-1, seq=sequence)
    models = ['obj_{:02d}'.format(sequence)]
    model_map = bench.models
    model = model_map[models[0]]
    
    print('models:', models[0])
    print('Diameter:', model.diameter)
    
    adds_t = []
    adds_d = []    
    
    for i in tqdm(range(gdt.shape[0])):
        gdt_pose_quat = gdt[i, :]
        pre_t_pose_quat = pre_triplet[i, :]
        pre_d_pose_quat = pre_depth[i, :]
        gdt_pose = Quaternion(gdt_pose_quat).rotation_matrix
        pre_t_pose = Quaternion(pre_t_pose_quat).rotation_matrix
        pre_d_pose = Quaternion(pre_d_pose_quat).rotation_matrix

        def transform_points(points_3d, mat):
            rot = np.matmul(mat, points_3d.transpose())
            return rot.transpose()

        gdt_tfm = transform_points(model.vertices, gdt_pose)
        pre_t_tfm = transform_points(model.vertices, pre_t_pose)
        pre_d_tfm = transform_points(model.vertices, pre_d_pose)

        g_t = np.mean(np.linalg.norm(gdt_tfm - pre_t_tfm, axis=1))
        g_d = np.mean(np.linalg.norm(gdt_tfm - pre_d_tfm, axis=1))

        adds_t.append(g_t < (0.15 * model.diameter))
        adds_d.append(g_d < (0.15 * model.diameter))

    ms_t = np.mean(adds_t)
    ms_d = np.mean(adds_d)
    
    print('matching score for triplet:', ms_t)
    print('matching score for depth:', ms_d)

    matching_score_file = os.path.join(basepath, 'matching_score.txt')
    matching_score = open(matching_score_file, 'w')
    matching_score.write(str('matching score for triplet:') + str(ms_t) + '\n')
    matching_score.write(str('matching score for depth:') + str(ms_d) + '\n')
    matching_score.close()