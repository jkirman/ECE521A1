# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 18:50:12 2018

@author: Jeffrey
"""

import tensorflow as tf
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import kNN
  
sess = tf.InteractiveSession()
init = tf.global_variables_initializer()
sess.run(init)

# 1D dataset
np.random.seed(521)
Data = np.linspace(1.0 , 10.0 , num =100) [:, np. newaxis]
Target = np.sin( Data ) + 0.1 * np.power( Data , 2) + 0.5 * np.random.randn(100 , 1)
randIdx = np.arange(100)
np.random.shuffle(randIdx)
trainData, trainTarget = Data[randIdx[:80]], Target[randIdx[:80]]
validData, validTarget = Data[randIdx[80:90]], Target[randIdx[80:90]]
testData, testTarget = Data[randIdx[90:100]], Target[randIdx[90:100]]

# Checking MSE of each portion of dataset
ks = [1,3,5,50]

mse_tra = np.zeros(4)
for i in range(0,4):
    mse_tra[i] = kNN.mse(kNN.k_NN(trainData, trainData, trainTarget, ks[i]), trainTarget).eval()

mse_val = np.zeros(4)
for i in range(0,4):
    mse_val[i] = kNN.mse(kNN.k_NN(validData, trainData, trainTarget, ks[i]), validTarget).eval()

mse_tes = np.zeros(4)
for i in range(0,4):
    mse_tes[i] = kNN.mse(kNN.k_NN(testData, trainData, trainTarget, ks[i]), testTarget).eval()

fig, ax = plt.subplots()
ax.plot( [1,3,5,50], mse_tra, 'ro',[1,3,5,50], mse_val, 'go',[1,3,5,50], mse_tes, 'bo')
plt.show()