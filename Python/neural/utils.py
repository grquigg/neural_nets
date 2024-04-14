import numpy as np
import struct
from array import array

def sigmoid(x):
    return 1/(1+np.exp(-x))

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)

def read_file(path) -> np.ndarray:
    inputs = None
    with open(path, 'rb') as file:
        magic = struct.unpack("4B", file.read(4))
        dims = struct.unpack(f">{magic[3]}I", file.read(4*magic[3]))
        arr = array("B", file.read())
        arr = np.array(arr)
        inputs = np.reshape(arr, dims)
    return inputs

def computeAccuracy(predicted, actual):
    correct = 0
    for i in range(predicted.shape[0]):
        m = 0
        max_score = 0.0
        a = 0
        for j in range(predicted.shape[1]):
            if (predicted[i][j] > max_score):
                m = j
                max_score = predicted[i][j]
            if (actual[i][j] == 1.0):
                a = j
        if (a == m):
            correct+=1
    return correct
