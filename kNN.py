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
    
def kNN_class(test_point, in_features, targets, k):
    distances = dis_euc(test_point, in_features)
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

def class_perf(results, targets): 
    error = tf.count_nonzero(results - targets) / tf.cast(tf.shape(targets), tf.int64)
    return tf.constant(1) - error

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