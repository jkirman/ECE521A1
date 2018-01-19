# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 19:57:54 2018

@author: Jeffrey
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import kNN_class

def data_segmentation(data_path, target_path, task):
# task = 0 >> select the name ID targets for face recognition task
# task = 1 >> select the gender ID targets for gender recognition task
    data = np.load(data_path)/255
    data = np.reshape(data, [-1, 32*32])
    
    target = np.load(target_path)
    
    np.random.seed(45689)
    rnd_idx = np.arange(np.shape(data)[0])
    np.random.shuffle(rnd_idx)
    
    trBatch = int(0.8*len(rnd_idx))
    validBatch = int(0.1*len(rnd_idx))
    
    trainData, validData, testData = data[rnd_idx[1:trBatch],:], \
    data[rnd_idx[trBatch+1:trBatch + validBatch],:],\
    data[rnd_idx[trBatch + validBatch+1:-1],:]
    
    trainTarget, validTarget, testTarget = target[rnd_idx[1:trBatch], task], \
    target[rnd_idx[trBatch+1:trBatch + validBatch], task],\
    target[rnd_idx[trBatch + validBatch + 1:-1], task]
    
    return trainData, validData, testData, trainTarget, validTarget, testTarget

def form_picture(data, index):
    pictures = np.reshape(data, [-1,32,32])
    return pictures[index]
    
sess = tf.InteractiveSession()
init = tf.global_variables_initializer()
sess.run(init)

(trainData, validData, testData, trainTarget, validTarget, testTarget) = data_segmentation("data.npy", "target.npy", 0)

classes = kNN_class.kNN_class(testData, trainData, trainTarget, 10)

#plt.imshow(form_picture(trainData,0), cmap="gray")
#plt.show()

#test = kNN_class.kNN_class(testData, trainData, trainTarget, 25, 0)