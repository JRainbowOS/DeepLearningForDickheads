# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 20:27:08 2019

@author: YS15101711
"""

import numpy as np

# Nonlinearity Function
def sigma(x):
    return 1 / (1 + np.exp(-x))

# And its derivative
def sigmaPrime(x):
    return x * (1 - x)


# Training Data
X = np.array([[0,1,0],
             [0,0,1],
             [1,0,0],
             [1,0,1]])

# Target Output
y = np.array([[1,1,0,0]]).T


# Initiate Weights
w0 = 2 * np.random.random((3,1)) - 1

# Augment Data
expFactor = 3
for i in range(expFactor):
    X = np.append(X, X)
    y = np.append(y, y)
X = X.reshape((2 ** expFactor) * 4, 3)    
y = y.reshape((2 ** expFactor) * 4, 1)


batch_size = 2 ** expFactor # 8
numSamples = X.shape[0] # 32
learningRate = 0.1

# Start Learning
for epoch in range(1000):
    # divide into batches
    for i in range(int(numSamples / batch_size)):
        l0 = X[i * batch_size: (i + 1) * batch_size, : ]
        
        # Forward Propagate
        l1 = sigma(np.dot(l0, w0))
        
        # Calculate basic error
        l1_error = y[i * batch_size: (i + 1) * batch_size] - l1

        # Calculate delta (error weighted derivative)
        l1_delta = l1_error * sigmaPrime(l1)
        
        # Update weights
        w0 += learningRate * np.dot(l0.T, l1_delta)


# Test Sample
X_test = np.array([0,1,1])

# Forward Propagate
y_test = sigma(np.dot(X_test, w0))

print(y_test)



