# -*- coding: utf-8 -*-
"""
Created on Fri Jun 30 11:47:54 2017

# 20170925: add depth images

@author: Yuanpeng
"""

from keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Input, Dense, Convolution2D, MaxPooling2D, UpSampling2D, Flatten, Reshape, Dropout, Lambda, Conv2D, Activation
from keras.layers.normalization import BatchNormalization
from keras.models import Model, Sequential
from keras.callbacks import TensorBoard
from keras.models import model_from_json
from keras import losses, optimizers
from keras import backend as K
from PIL import Image
import PIL as pit
import sys
import os
import cv2
import random
from random import shuffle, randint, choice
#from ComputeDistanceMap import ComputeDistanceMap

def create_network(input_shape):
    '''Base network to be shared (eq. to feature extraction).
    '''
    seq = Sequential()
    seq.add(Conv2D(16, (3, 3), data_format="channels_last", input_shape=input_shape, padding='same', activation='relu'))
    seq.add(MaxPooling2D((2, 2), data_format="channels_last", padding="same"))
    seq.add(Conv2D(8, (3, 3), data_format="channels_last", padding='same', activation='relu'))
    seq.add(MaxPooling2D((2, 2), data_format="channels_last", padding="same"))
    seq.add(Conv2D(8, (3, 3), data_format="channels_last", padding='same', activation='relu'))
    seq.add(MaxPooling2D((2, 2), data_format="channels_last", padding="same"))
#    seq.add(Dense(2000, input_shape=input_shape, activation='relu'))
#    seq.add(Dropout(0.1))
#    seq.add(Dense(1000, activation='relu'))
#    seq.add(Dropout(0.1))
    seq.add(Flatten())
    seq.add(Dense(500, activation='relu'))
    #seq.add(BatchNormalization())
    seq.add(Dropout(0.4))       #0.1
    seq.add(Dense(1000, activation='relu'))
    #seq.add(BatchNormalization())
    seq.add(Dropout(0.4))       #0.1
    seq.add(Dense(18, activation='relu'))
    #seq.add(BatchNormalization())
    #seq.add(Activation('relu'))
    return seq

def normalization(x):
    length = K.sqrt(K.maximum(K.sum(x ** 2, axis=1, keepdims=True), K.epsilon()))
    return x / length
    
def normal_output_shape(shape):
    return (shape[0], shape[1])
    
def feature_divide(vects):
    x, y = vects
    return x / y
    
def fea_div_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)
    
def euclidean_distance(vects):
    x, y = vects
    return K.sqrt(K.maximum(K.sum(K.square(x - y), axis=1, keepdims=True), K.epsilon())) 
    #return K.sqrt(K.square(x - y))
    
def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)
    
def contrastive_loss(y_true, y_pred):
    '''Contrastive loss from Hadsell-et-al.'06
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    '''
    margin = 1
    return K.mean(y_true * K.square(y_pred) +
                  (1 - y_true) * K.square(K.maximum(margin - y_pred, 0)))
    
def loss_triplet_1(y_true, y_pred):
    return K.mean(K.square(y_true - y_pred), axis=1, keepdims=True)
    
def loss_triplet_2(y_true, y_pred):
    margin = 1
    return K.sum(y_true * K.maximum(margin - y_pred, 0) + (1 - y_true) * K.maximum(y_pred - margin, 0))
    
def creat_pairs(img, pose, n):
    pairs = []
    labels = []

    idx_1 = np.random.choice(img.shape[0], n)
    idx_2 = np.random.choice(img.shape[0], n)
    for i in range(n):
        idx1, idx2 = idx_1[i], idx_2[i]
        data1, data2 = img[idx1], img[idx2]
        pairs += [[data1, data2]]
        dis = np.sqrt(np.dot((pose[idx1] - pose[idx2]), (pose[idx1] - pose[idx2]).T))
        #dis = np.e ** dis - 1
        labels += [dis]

    return np.array(pairs).reshape((-1,2,48,48,1)), np.array(labels), idx_1, idx_2
    
def _creat_pairs(img, pose):
    pairs = []
    labels = []

    img_bench = img[0]
    pose_bench = pose[0]

    for i in range(img.shape[0]):
        data = img[i]
        pairs += [[img_bench, data]]
        dis = np.sqrt(np.dot((pose[i] - pose_bench), (pose[i] - pose_bench).T))
        labels += [dis]

    return np.array(pairs).reshape((-1,2,48,48,1)), np.array(labels), idx_1, idx_2
    
def creat_triplets(img, pose, n, channel):
    triple_set = []
    labels = []

    idx_1 = np.random.choice(img.shape[0], n)
    idx_2 = np.random.choice(img.shape[0], n)
    idx_3 = np.random.choice(img.shape[0], n)
    
    for i in range(n):
        idx_anchor, idx1, idx2 = idx_1[i], idx_2[i], idx_3[i]
        if idx_anchor == idx2:
            idx2 += 1
        data_anchor, data1, data2 = img[idx_anchor], img[idx1], img[idx2]
        triple_set += [[data_anchor, data1, data2]]
        dis1 = np.sqrt(np.dot((pose[idx_anchor] - pose[idx1]), (pose[idx_anchor] - pose[idx1]).T))
        dis2 = np.sqrt(np.dot((pose[idx_anchor] - pose[idx2]), (pose[idx_anchor] - pose[idx2]).T))
        '''
        if dis2 > 1e-5:
            label = dis1 / dis2
        else:
            label = 1e6
        '''    
        if dis1 >= dis2:
            label = 1
        else:
            label = 0
        
        labels += [label]
    
    return np.array(triple_set).reshape((-1,3,48,48,channel)), np.array(labels), idx_1, idx_2, idx_3
    
def compute_accuracy(predictions, labels):
    '''Compute classification accuracy with a fixed threshold on distances.
    '''
    m, pr = 0, predictions.ravel()
    for i in range(pr.shape[0]):
        dis = np.sqrt(np.dot((pr[i] - labels[i]), (pr[i] - labels[i]).T))
        if dis < 0.01:
            m += 1
    return m/pr.shape[0]

def train_pairs_model(model, img_train, img_test, label_train, label_test):
    print('Training pair model...')
    sgd = optimizers.SGD(lr=0.3)
    model.compile(optimizer=sgd, loss = loss_triplet_1, metrics=['accuracy'])
    model.fit([img_train[:,0], img_train[:,1]], label_train, epochs=200, batch_size=128, shuffle=True, callbacks=[TensorBoard(log_dir='/tmp/tb', histogram_freq=0, write_graph=False)])
    
    print("Evaluating model...")
    score = model.evaluate([img_train[:,0], img_train[:,1]], label_train, verbose=0)                    # pairs
    print ("%s: %.2f%%" % (model.metrics_names[1], score[1]*100))
    
    pred = model.predict([img_train[:,0], img_train[:,1]])
    tr_acc = compute_accuracy(pred, label_train)
    pred = model.predict([img_test[:,0], img_test[:,1]])
    te_acc = compute_accuracy(pred, label_test)
    print('* Accuracy on training set: %0.2f%%' % (100 * tr_acc))
    print('* Accuracy on test set: %0.2f%%' % (100 * te_acc))
    
def train_triple_model(model, img_train, img_test, label_train, label_test):
    print('Training triplet model...')
    sgd = optimizers.SGD(lr=0.1)       #for triplet 2, 0.03 is good
    model.compile(optimizer=sgd, loss = loss_triplet_2, metrics=['accuracy'])
    model.fit([img_train[:,0], img_train[:,1], img_train[:,2]], label_train, epochs=400, batch_size=128, shuffle=True, callbacks=[TensorBoard(log_dir='/tmp/tb', histogram_freq=0, write_graph=False)]) # triple sets
    
    print("Evaluating model...")
    score = model.evaluate([img_train[:,0], img_train[:,1], img_train[:,2]], label_train, verbose=0)   # triple sets
    print ("%s: %.2f%%" % (model.metrics_names[1], score[1]*100))
    
    pred = model.predict([img_train[:,0], img_train[:,1], img_train[:,2]])
    tr_acc = compute_accuracy(pred, label_train)
    pred = model.predict([img_test[:,0], img_test[:,1], img_test[:,2]])
    te_acc = compute_accuracy(pred, label_test)
    print('* Accuracy on training set: %0.2f%%' % (100 * tr_acc))
    print('* Accuracy on test set: %0.2f%%' % (100 * te_acc))

def save_model(model):
    print("Saving model...")
    model_json = model.to_json()
    with open(base_dir + obj_name + "/model_triplet.json", "w") as json_file:
        json_file.write(model_json)
        
    print("Saving weights...")
    model.save_weights(base_dir + obj_name + "/model_triplet.h5")
    
# Prepares data
#x_train, x_test, y_train, y_test = [], [], [], []
base_dir = './dataset/Images'
obj_name = '/Ape2'
depth = False
depth_only = False
channel = 1
'''
total_array_grey = []

for imgs in os.listdir(base_dir + obj_name + binary_dir):                       #r"../Output3/Binary"
    img = Image.open(os.path.join(base_dir + obj_name + binary_dir,imgs))       #'../Output3/Binary'
    img_grey = img.convert('L')
    img_grey_array = np.array(img_grey.resize((48,48)))
    
    total_array_grey.append(img_grey_array.astype('float32')/255.)

imgs = np.array(total_array_grey)
idx_testing = np.random.choice(imgs.shape[0], 1000, replace = False)
testing_set = imgs[idx_testing]
idx_training = np.array(list(set(np.arange(imgs.shape[0])) - set(idx_testing)))
training_set = imgs[idx_training]
# imgs = np.loadtxt('../Output3/imgs.txt')
poses = np.loadtxt(base_dir + obj_name + '/poses.txt')                                                #'../Output3/poses.txt'
'''
imgs_train = np.load(base_dir + obj_name + '/imgs_train.npy')
poses_train = np.load(base_dir + obj_name + '/poses_train.npy')
imgs_test = np.load(base_dir + obj_name + '/imgs_test.npy')
poses_test = np.load(base_dir + obj_name + '/poses_test.npy')

#imgs_train.resize((imgs_train.shape[0], 48, 48))
#imgs_test.resize((imgs_test.shape[0], 48, 48))

if depth:
    depth_train = np.load(base_dir + obj_name + '/depths_train.npy')
    depth_test = np.load(base_dir + obj_name + '/depths_test.npy')
    #depth_train.resize((depth_train.shape[0], 48, 48))
    #depth_test.resize((depth_test.shape[0], 48, 48))

    if depth_only:
        imgs_train = depth_train
        imgs_test = depth_test
    else:
        channel = 2
        train = np.zeros((imgs_train.shape[0], 48, 48, 2))
        test = np.zeros((imgs_test.shape[0], 48, 48, 2))
        for i in range(imgs_train.shape[0]):
            train[i,:,:,0] = imgs_train[i]
            train[i,:,:,1] = depth_train[i]
        for i in range(imgs_test.shape[0]):
            test[i,:,:,0] = imgs_test[i]
            test[i,:,:,1] = depth_test[i]
        imgs_train = train
        imgs_test = test

#pairs, labels, idx_1, idx_2 = creat_pairs(imgs_train, poses_train, 50000)        # pairs
pairs, labels, idx_1, idx_2, idx_3 = creat_triplets(imgs_train, poses_train, 50000, channel)   # triple sets             #imgs

img_train = pairs[:40000]
img_test = pairs[40000:]
label_train = labels[:40000]
label_test = labels[40000:]

# Creates the Convolutional Auto Encoder 
input_shape = (48,48,channel)
input_1 = Input(shape=input_shape)
input_2 = Input(shape=input_shape)
input_3 = Input(shape=input_shape)          # triple sets

net = create_network(input_shape)
x1 = net(input_1)
x2 = net(input_2)
x3 = net(input_3)                           # triple sets

x1 = Lambda(normalization, output_shape=normal_output_shape)(x1)      #, output_shape=normal_output_shape
x2 = Lambda(normalization, output_shape=normal_output_shape)(x2)
x3 = Lambda(normalization, output_shape=normal_output_shape)(x3)


#conv1 = Convolution2D(8, 3, 3, activation='relu', border_mode='same', dim_ordering='tf')
#pooling1 = MaxPooling2D((2, 2), border_mode='same', dim_ordering='tf')
#dense1 = Dense(1000)
#dense2 = Dense(20)
#
#x1 = conv1(input_img)
#x2 = conv1(input_bench)
#x1 = pooling1(x1)
#x2 = pooling1(x2)
#x1 = Flatten()(x1)
#x2 = Flatten()(x2)
#x1 = dense1(x1)
#x2 = dense1(x2)
#x1 = dense2(x1)
#x2 = dense2(x2)

dis_feas = Lambda(euclidean_distance, output_shape=eucl_dist_output_shape)([x1, x2])

dis_feas_2 = Lambda(euclidean_distance, output_shape=eucl_dist_output_shape)([x1, x3])      # triple sets
k = Lambda(feature_divide, output_shape=fea_div_output_shape)([dis_feas, dis_feas_2])                                                                   # triple sets

print("Creating model...")
model_fea = Model(inputs = input_1, outputs = x1)
model_dis = Model(inputs = [input_1, input_2], outputs = dis_feas)
#train_pairs_model(model_dis, img_train, img_test, label_train, label_test)      # pairs
model = Model(inputs = [input_1, input_2, input_3], outputs = k)          # triple sets
train_triple_model(model, img_train, img_test, label_train, label_test)    # triple sets

#save_model(model_dis)               # pairs
#save_model(model)                  # triple sets
save_model(model_fea)

print("Saving features...")
training_features = model_fea.predict(imgs_train.reshape((-1,48,48,channel)))
np.savetxt(base_dir + obj_name + '/training_features_triplet.txt', training_features)                                                              #'../Output3/features.txt'
testing_features = model_fea.predict(imgs_test.reshape((-1,48,48,channel)))
np.savetxt(base_dir + obj_name + '/testing_features_triplet.txt', testing_features)                                                              #'../Output3/features.txt'
'''
print("Saving index...")
np.savetxt(base_dir + obj_name + '/training_index.txt', idx_training) 
np.savetxt(base_dir + obj_name + '/testing_index.txt', idx_testing) 
'''
print("Saving distances...")
distances = model_dis.predict([pairs[:,0], pairs[:,1]])
np.savetxt(base_dir + obj_name + '/distances_triplet.txt', distances)                                                             #'../Output3/distances.txt'

# triple set
print("Saving times...")
times = model.predict([pairs[:,0], pairs[:,1], pairs[:,2]])
np.savetxt(base_dir + obj_name + '/times_triplet.txt', times)                                                                 #'../Output3/times.txt'

#print('Show result')
#ComputeDistanceMap(poses_train, training_features, base_dir + obj_name)
# Tests the model and shows results
#x_test = x_test[:10]
#y_test = y_test[:10]
#
#print("Making predictions...")
#predictions = autoencoder.predict(x_test)
#x_test = [cv2.cvtColor(img,cv2.COLOR_GRAY2RGB) for img in x_test]
#predictions = [cv2.cvtColor(img,cv2.COLOR_GRAY2RGB) for img in predictions]
#
#img = np.vstack((np.hstack(x_test),np.hstack(predictions)))
#
#cv2.imshow("Input - Reconstructed - Ground truth",cv2.resize(img,(img.shape[1],img.shape[0])))
#cv2.waitKey(0)
#cv2.destroyAllWindows()