# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 10:52:36 2018

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

x = np.linspace(0.0,11.0,num = 1000)[:,np.newaxis]
y_hat1 = kNN.k_NN(x, trainData, trainTarget, 1)
y_hat3 = kNN.k_NN(x, trainData, trainTarget, 3)
y_hat5 = kNN.k_NN(x, trainData, trainTarget, 5)
y_hat50 = kNN.k_NN(x, trainData, trainTarget, 50)


fig, ax = plt.subplots()
ax.plot( x, tf.transpose(y_hat1).eval(), 'r-', x, tf.transpose(y_hat3).eval(), 'b-', x, tf.transpose(y_hat5).eval(), 'g-', x, tf.transpose(y_hat50).eval(), 'y-', Data, Target, 'o')
plt.show()