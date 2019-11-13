import numpy as np

def relu(X):
    return np.maximum(X, 0)

def relu_deriv(X):
    return 1. * (X > 0)

def softmax(X):
    expX = np.exp(X)
    return expX / expX.sum(axis=0, keepdims=True)
    