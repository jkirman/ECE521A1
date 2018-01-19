# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 20:29:03 2018

@author: Jeffrey
"""

import tensorflow as tf
import numpy as np
import kNN

def dis_euc_mat(X1, X2):
# Takes in two tensors X1, X2 of size (N1,D,D) and (N2,D,D) respectively
# and calculates the euclidean distances between them
#
# Reshapes the input vectors into 4D vectors which are broadcasted
#   onto eachother for subtraction operation, which is then piece-wise
#   squared and finally the extra added dimension is reduction-summed
    length = tf.shape(X1)[2]
    X1_new = tf.reshape(X1, [tf.shape(X1)[0], 1, length, length] )
    X2_new = tf.reshape(X2, [1, tf.shape(X2)[0], length, length] )
    distances = tf.reduce_sum(tf.reduce_sum((X1_new - X2_new)**2, 3), 2)
    return distances

#def kNN_class(test_point, in_features, targets, k, cls):
#    distances = dist_euc_mat(test_point, in_features)
#    (val, ind) = tf.nn.top_k(-distances,k) # find closest neighbours in training set
#    candidates = tf.gather(targets[cls],ind) # Find the classifications for these neighbours
#    length = tf.shape(candidates)[0]
#    class_list = []
#    count_list = []

#def kNN_class(test_point, in_features, targets, k):
#    distances = kNN.dis_euc(test_point, in_features)
#    (val, ind) = tf.nn.top_k(-distances,k) # find closest neighbours in training set
#    
#    candidates = tf.gather(tf.constant(targets),ind) # Find the classifications for these neighbours
#    length = tf.shape(candidates)[0]
#    class_list = []
#    count_list = []
#
#    # Count the frequency of nearest neighbours and put them into matrices
#    # which are reduced class list and count list
#    for i in range(0,length.eval()):
#        (temp_class, __, temp_count) = tf.unique_with_counts(candidates[i])
#        padding = tf.concat([tf.constant([0]), tf.constant([k]) - tf.shape(temp_class)],0)
#        print(padding.eval())
#        class_list.append(tf.pad(temp_class,padding))
#        count_list.append(tf.pad(temp_count,padding))
#        print(class_list.eval())
#    
#    red_class_list = tf.stack(class_list)
#    red_count_list = tf.stack(count_list)
#    
#    # Create an iterator for each test_point
#    iterator = tf.cast(tf.linspace(0., length.eval() - 1., length.eval()), tf.int64)
#    iterator = tf.reshape(iterator,[length,1])
#
#    # Combine the iterator with the indices of the highest counts in the
#    # reduced count list (red_count_list)
#    count_loc = tf.concat([iterator, tf.reshape(tf.argmax(red_count_list,1),[length,1])],1)
#    
#    outputs = tf.gather_nd(red_class_list, count_loc)
#
#    return outputs
#    return classifier[tf.argmax(count).eval()]

def kNN_class(test_point, in_features, targets, k):
    distances = kNN.dis_euc(test_point, in_features)
    (val, ind) = tf.nn.top_k(-distances,k) # find closest neighbours in training set
    
    candidates = tf.gather(tf.constant(targets),ind) # Find the classifications for these neighbours
    length = tf.shape(candidates)[0]
    class_list = []
    count_list = []

    # Count the frequency of nearest neighbours and put them into matrices
    # which are reduced class list and count list
    for i in range(0,length.eval()):
        (temp_class, __, temp_count) = tf.unique_with_counts(candidates[i])
        padding = tf.concat([tf.constant([0]), tf.constant([k]) - tf.shape(temp_class)],0)
        class_list.append(tf.pad(temp_class,[padding]))
        count_list.append(tf.pad(temp_count,[padding]))
    
    red_class_list = tf.stack(class_list)
    red_count_list = tf.stack(count_list)
    
    # Create an iterator for each test_point
    iterator = tf.cast(tf.linspace(0., length.eval() - 1., length.eval()), tf.int64)
    iterator = tf.reshape(iterator,[length,1])

    # Combine the iterator with the indices of the highest counts in the
    # reduced count list (red_count_list)
    count_loc = tf.concat([iterator, tf.reshape(tf.argmax(red_count_list,1),[length,1])],1)
    
    outputs = tf.gather_nd(red_class_list, count_loc)

    return outputs
                       
#a = tf.constant([[[0,1], [-1,0]], [[2,0], [0,2]], [[3,-3], [-3,-3]]])
#b = tf.constant([[[5,0],[4,0]], [[3,3],[3,3]], [[1,0],[2,0]], [[2,2],[-2,-2]]], name='b')
#c = kNN_class(a,b,[[1,2,3,4],[5,6,7,8]],2,1)
#
#sess = tf.InteractiveSession()
#init = tf.global_variables_initializer()
#sess.run(init)
#
#print(sess.run([c]))