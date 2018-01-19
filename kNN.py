# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 10:51:40 2018

@author: Jeffrey
"""

import tensorflow as tf

## Q1: Euclidian distance function ##
def dis_euc(X1, X2):
# Takes in two tensors X1, X2 of size (N1,D) and (N2, D) respectively
# and calculates the euclidean distances between them
#
# Reshapes the input vectors into 3D vectors which are broadcasted
#   onto eachother for subtraction operation, which is then piece-wise
#   squared and finally the extra added dimension is reduction-summed
    length = tf.shape(X1)[1]
    X1_new = tf.reshape(X1, [tf.shape(X1)[0], 1, length] )
    X2_new = tf.reshape(X2, [1, tf.shape(X2)[0], length] )
    distances = tf.reduce_sum((X1_new - X2_new)**2, axis = 2)
    return distances

def nn_resp(distances, k):
    size = tf.shape(distances)[1]
    k = min(k, size.eval())
    (val, ind) = tf.nn.top_k(-distances,k)
    resp = tf.one_hot(ind, size)
    r_star = tf.reduce_sum(resp,1) / k
    return r_star
    
def k_NN(test_point, in_features, targets, k):
    r_star = tf.cast(nn_resp(dis_euc(test_point, in_features), k), tf.float64)
    targs = tf.constant(targets, tf.float64)
    return tf.matmul(targs, r_star, True, True)

def mse(predicted, targets):
    error_sum = tf.reduce_sum((predicted - targets) ** 2)
    size = tf.cast(tf.shape(predicted)[1], tf.float64)
    return error_sum / (2*size)

##a = tf.constant([[1,1],[2,2],[3,3],[4,4]], name='a')
#a = tf.constant([[1,1], [2,2]])
#b = tf.constant([[-1,6],[1,7],[2,5],[8,8]], name='b')
#c = dis_euc(a,b)
#d = nn_resp(c,2);
#
#
#sess = tf.InteractiveSession()
#init = tf.global_variables_initializer()
#sess.run(init)
#
##print([a.shape,b.shape])
#print(d)
#print(sess.run([c,d]))