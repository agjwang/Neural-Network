import numpy as np


def sigmoid(Z):
    A = 1 / (1 + np.exp(-Z))

    assert(A.shape == Z.shape)
    return A


def relu(Z):
    A = np.maximum(0, Z)

    assert(A.shape == Z.shape)
    return A


def tanh(Z):
    A = (np.exp(Z) - np.exp(-Z)) / (np.exp(Z) + np.exp(-Z))

    assert(A.shape == Z.shape)
    return A
