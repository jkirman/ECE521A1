# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 20:29:03 2018

@author: Jeffrey
"""

import tensorflow as tf

def euc_dist_mat(X1, X2):
# Takes in two tensors X1, X2 of size (D,D,N1) and (D,D,N2) respectively
# and calculates the euclidean distances between them
#
# Reshapes the input vectors into 4D vectors which are broadcasted
#   onto eachother for subtraction operation, which is then piece-wise
#   squared and finally the extra added dimension is reduction-summed