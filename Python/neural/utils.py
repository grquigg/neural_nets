import numpy as np
import struct
from array import array

def cross_entropy_loss(y_true, y_pred):
    # Small epsilon to prevent log(0) scenario
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    # Compute the cross entropy for each row (observation)
    ce_loss = -np.sum(y_true * np.log(y_pred), axis=1)
    # Average over all observations
    return np.mean(ce_loss)

def sigmoid(x):
    return 1/(1+np.exp(-x))

def sigmoid_derivative(x):
    return np.multiply(sigmoid(x), (1 - sigmoid(x)))

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)

def relu(x) -> np.ndarray:
    return np.maximum(0, x)

def relu_derivative(x) -> np.ndarray:
    return (x > 0).astype(float)


def get_derivative_for_activation_fn(fn):
    if fn == relu:
        return relu_derivative
    elif fn == sigmoid:
        return sigmoid_derivative
    
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
