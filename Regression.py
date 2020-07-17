# -*- coding: utf-8 -*-
"""
Created on Thu Aug 10 16:53:36 2017

@author: Yuanpeng
"""

import tensorflow as tf
import numpy as np
import os
from matplotlib import pyplot as plt
from CheckQuaternions import DrawQuaternions

def weight_variable(shape):
    initial = tf.truncated_normal(shape,stddev=0.1) # 截断正态分布，此函数原型为尺寸、均值、标准差
    return tf.Variable(initial)
    
def bias_variable(shape):
    initial = tf.constant(0.1,shape=shape)
    return tf.Variable(initial)
    
def conv2d(x,W):    # 这里不使用这个函数
    return tf.nn.conv2d(x,W,strides=[1,1,1,1], padding='SAME') # strides第0位和第3为一定为1，剩下的是卷积的横向和纵向步长
    
def maxpooling2d(x):    # 向量不使用池化，图像使用
    return tf.nn.max_pool(x,ksize = [1,2,2,1],strides=[1,2,2,1], padding='SAME')   # 参数同上，ksize是池化块的大小

def CreatNetwork(x=None, depth=2, keep_prob=None):
    if depth < 1:
        print('network needs at least two layers')
        return x
    hidden_node = np.array([120, 120, 120, 120, 120, 120])
    w_fc = tf.TensorArray(tf.float32, size=depth)
    b_fc = tf.TensorArray(tf.float32, size=depth)
    #w_fc = np.zeros((depth,))
    #b_fc = np.zeros((depth,))
    for i in range(0, depth):
        if i == 0:
            node1 = 20
        else:
            node1 = hidden_node[i-1]
        if i == (depth-1):
            node2 = 4
        else:
            node2 = hidden_node[i]
        w_fc[i] = weight_variable([node1, node2])
        b_fc[i] = bias_variable([node2])
    fc = tf.nn.tanh(tf.matmul(x, w_fc[0]) + b_fc[0])
    dropout = tf.nn.dropout(fc, keep_prob)
    for i in range(1, depth):
        fc = tf.nn.tanh(tf.matmul(dropout, w_fc[i]) + b_fc[i])
        if i < (depth-1):
            dropout = tf.nn.dropout(fc, keep_prob)
    return fc

# 导入数据     
base_dir = '../Images'

#obj_dict = {'ape':'Ape2',   'benchviseblue':'Benchviseblue2',   'bowl':'Bowl2',     'camera':'Camera2', 
#            'can':'Can2',   'cat':'Cat2',                       'cup':'Cup2',       'drill':'Drill2', 
#            'duck':'Duck2', 'eggbox':'Eggbox2',                 'glue':'Glue2',     'holepuncher':'Holepuncher2', 
#            'iron':'Iron2', 'lamp':'Lamp2',                     'phone':'Phone2',   'elephent':'Elephent',
#            'my_bulldog':'My_Bulldog', 'my_cat':'My_Cat', 'my_dog':'My_Dog', 'my_leopard':'My_Leopard',
#            'my_pig':'My_Pig'}
ftype='_triplet'
#for key in obj_dict:
#    obj_name = obj_dict[key]
#    print('Now processing', obj_name)
#    folder = os.path.join(base_dir, obj_name)
#    if not os.path.exists(folder):
#        print('Missing folder', obj_name)
#        continue
#    if obj_name == 'Holepuncher2' or obj_name == 'Lamp2' or obj_name =='Phone2':
#        print('Jumping folder', obj_name)
#        continue
obj_name = 'Eggbox2'
print('Now processing', obj_name)
model_type = 'regression' + ftype
basepath = os.path.join(base_dir, obj_name)
modelpath = os.path.join(basepath,  model_type)
if not os.path.exists(modelpath):
    os.mkdir(modelpath)
training_features = np.loadtxt(os.path.join(basepath, 'training_features' + ftype + '.txt'))
testing_features = np.loadtxt(os.path.join(basepath, 'testing_features' + ftype + '.txt'))
poses_train = np.load(os.path.join(basepath, 'poses_train.npy'))
poses_test = np.load(os.path.join(basepath, 'poses_test.npy'))

codes = training_features
poses = poses_train

#codes = np.loadtxt(base_dir + obj_name + '/training_features.txt')
#poses = np.loadtxt(base_dir + obj_name + '/poses.txt')

if codes.shape[0] != poses.shape[0]:
    exit()

# 验证
codenum = codes.shape[0]
codeidx = np.random.permutation(range(codenum))
codes = codes[codeidx,:]
poses = poses[codeidx,:]
print(poses[int(np.argwhere(codeidx==0))])

# 准备数据
training_x = codes[:4000, :]            #4500
training_y = poses[:4000, :]
test_x = codes[4000:, :]
test_y = poses[4000:, :]

# 参数设置
input_diam = 18
it_num = 5000
kp1 = 0.7
kp2 = 0.7
hidden_node = np.array([300, 0, 0, 0, 0]) #120, 300, 40
# Construct model

x = tf.placeholder(tf.float32, shape=[None, input_diam], name='x')
y = tf.placeholder(tf.float32, shape=[None, 4], name='y')
keep_prob = tf.placeholder(tf.float32, name='keep_prob')

'''
hidden_node = 250  #30-150
hidden_node_1 = 120
'''
#hidden_node_2 = 120

#w_conv1 = weight_variable([3, 3, 2, 16])
#w_conv2 = weight_variable([3, 3, 16, 32])
#w_conv3 = weight_variable([3, 3, 32, 32])
#b_conv1 = bias_variable([16])
#b_conv2 = bias_variable([32])
#b_conv3 = bias_variable([32])

w_fc1 = weight_variable([input_diam, hidden_node[0]])
w_fc2 = weight_variable([hidden_node[0], 4])
#w_fc3 = weight_variable([hidden_node[1], hidden_node[2]])
#w_fc4 = weight_variable([hidden_node[2], 4])
#w_fc5 = weight_variable([hidden_node[3], 4])

b_fc1 = bias_variable([hidden_node[0]])
b_fc2 = bias_variable([4])
#b_fc3 = bias_variable([hidden_node[2]])
#b_fc4 = bias_variable([4])
#b_fc5 = bias_variable([4])

#fold = tf.reshape(x, shape=[-1, 8, 8, 2])
#conv1 = tf.nn.relu(tf.nn.conv2d(fold, w_conv1, strides=[1, 1, 1, 1], padding='SAME') + b_conv1)
#pooling1 = tf.nn.max_pool(conv1, ksize = [1,2,2,1],strides=[1,2,2,1],padding='SAME')
#conv2 = tf.nn.relu(tf.nn.conv2d(conv1, w_conv2, strides=[1, 1, 1, 1], padding='SAME') + b_conv2)
#pooling2 = tf.nn.max_pool(conv2, ksize = [1,2,2,1],strides=[1,2,2,1],padding='SAME')
#conv3 = tf.nn.relu(tf.nn.conv2d(pooling2, w_conv3, strides=[1, 1, 1, 1], padding='SAME') + b_conv3)
#unfold = tf.reshape(conv2, shape=[-1, 2048])
#fc = tf.nn.relu(tf.matmul(x, w_fc1) + b_fc1)
fc = tf.nn.tanh(tf.matmul(x, w_fc1) + b_fc1)
dropout = tf.nn.dropout(fc, keep_prob)

#fc_2 = tf.nn.tanh(tf.matmul(dropout, w_fc2) + b_fc2)
#dropout = tf.nn.dropout(fc_2, keep_prob)

#fc_3 = tf.nn.tanh(tf.matmul(dropout, w_fc3) + b_fc3)
#dropout = tf.nn.dropout(fc_3, keep_prob)

#y_hat = tf.nn.relu(tf.matmul(dropout, w_fc2) + b_fc2)
y_hat = tf.nn.tanh(tf.matmul(dropout, w_fc2) + b_fc2, name='y_hat')
#y_hat = tf.maximum(0., y_hat)
#y_hat = CreatNetwork(x, 2, keep_prob)
#loss = tf.reduce_mean(tf.square(y - y_hat))     #l2_loss
'''
flag = tf.cast(y_hat[:,:2] > 0, dtype = tf.float32)
con_one = tf.Variable(tf.constant(1.0, shape = [100]))
flag_quaternion = tf.Variable([flag[:,0], flag[:,1], con_one, con_one])
flag_quaternion = tf.transpose(flag_quaternion)
y_hat = tf.multiply(flag_quaternion, y_hat)
'''
loss = tf.reduce_mean(tf.square(y - y_hat))
#loss = tf.reduce_sum(tf.square(y - y_hat))

training = tf.train.AdamOptimizer(1e-4).minimize(loss)      #Grreduce_suzadientDescentOptimizer

acct_mat = tf.equal(tf.argmax(y_hat, 1), tf.argmax(y, 1))
acct_res = tf.reduce_mean(tf.cast(acct_mat, tf.float32))
error = tf.subtract(tf.constant(1.0), acct_res)

# Saver definition
model_saver = tf.train.Saver()

# Main

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

var = np.zeros((it_num,1))
error_training = np.zeros((it_num,1))
error_testing = np.zeros((it_num,1))
loss_training = np.zeros((it_num,1))

test_z = np.ones(test_y.shape)
test_z[:,:2] = (test_y[:,:2]>0).astype('float32')
flag_z = np.ones(poses_test.shape)
flag_z[:,:2] = (poses_test[:,:2]>0).astype('float32')

for i in range(it_num):
    idx = np.random.choice(training_x.shape[0], 100, replace = False)
    batch_x = training_x[idx, :]
    batch_y = training_y[idx, :]
    batch_z = np.ones(batch_y.shape)
    batch_z[:,:2] = (batch_y[:,:2]>0).astype('float32')
    var[i, 0 ] = i
    _, error_training[i, 0], loss_training[i, 0] = sess.run([training, error, loss], feed_dict={x:batch_x, y:batch_y, keep_prob: kp1}) #0.9
    if i%100 == 0:
        print('accuracy:', 1 - error_training[i, 0])
    error_testing[i, 0] = sess.run(error, feed_dict={x:test_x, y:test_y, keep_prob: kp1}) #1.0

y_hat, acct_res = sess.run([y_hat, acct_res], feed_dict = {x:testing_features, y:poses_test, keep_prob:kp1}) #1.0      #{x: codes, y :poses, keep_prob: kp2}

model_saver.save(sess, os.path.join(modelpath, 'regression_model' + ftype), global_step=it_num)

print ('test accuracy: ', acct_res) #1.0

acc_file_path = os.path.join(basepath, 'reg_acc' + ftype + '.txt')
acc_file = open(acc_file_path, 'w')
acc_file.write(str('acct_res: ') + str(acct_res) + '\n')
acc_file.close()

fg_1=plt.figure('error_training')
plt.plot(var.tolist(), error_training.tolist(), 'r', label='error_training')
plt.savefig('error_training' + ftype + '.png', dpi = 120)
fg_2=plt.figure('loss_training')
plt.plot(var.tolist(), loss_training.tolist(), 'b', label='loss_training')
plt.savefig('loss_training' + ftype + '.png', dpi = 120)
fg_3=plt.figure('error_testing')
plt.plot(var.tolist(), error_testing.tolist(), 'g', label='error_testing')
plt.savefig('error_testing' + ftype + '.png', dpi = 120)
#plt.show()

np.savetxt(os.path.join(basepath, 'pre_quaternions' + ftype + '.txt'), y_hat)
np.savetxt(os.path.join(basepath, 'gdt_quaternions' + ftype + '.txt'), poses_test)

prefix = str(hidden_node[0]) + '_' + str(hidden_node[1]) + '_' + str(hidden_node[2]) + '_' + str(hidden_node[3]) + '_' + str(hidden_node[4]) + '_' + 'kp1-' + str(kp1) + '_' + 'kp2-' + str(kp2) + '_' + 'dim-' + str(input_diam) + '_' + 'it-' + str(it_num)
print('paras: ' + prefix)
DrawQuaternions(poses_test, y_hat, basepath=basepath, ftype=ftype, prefix=prefix)

sess.close()
'''  
    for i in dir():
        if i == 'base_dir' or i == 'obj_dict' or i == 'ftype' or i == 'os' or i == 'plt' or i == 'tf' or i == 'np' or i == 'DrawQuaternions' or i == 'weight_variable' or i == 'bias_variable' or i == 'conv2d' or i == 'maxpooling2d' or i == 'CreatNetwork' or i.startswith('_'):
            continue
        #if i.startswith('test') or i.startswith('train') or i.startswith('obj') or i.startswith('ba') or i.startswith('code') or i.startswith('error') or i.startswith('fg') or i.startswith('pose') or i.startswith('model') or i.startswith('y') or i.startswith('var'):
        locals().pop(i)
'''
