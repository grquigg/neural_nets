import struct
from array import array
import numpy as np
from scipy.special import softmax

def read_file(path):
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

if __name__ == "__main__":
    input_path = "./MNIST_ORG/train-images.idx3-ubyte"
    output_path = "./MNIST_ORG/train-labels.idx1-ubyte"
    test_size = 40
    n_features = 20  # example value
    out_classes = 5  # example value
    BATCH_SIZE = 20
    learning_rate = 0.01
    epochs = 100
    # Initialize inputs array with zeros and then set specific indices to 1.0
    inputs = np.zeros((test_size, n_features), dtype=float)
    for i in range(test_size):
        inputs[i, i % n_features] = 1.0
    print(inputs)
    # Initialize outputs array with zeros and then set specific indices to 1.0
    outputs = np.zeros((test_size, out_classes), dtype=float)
    for i in range(test_size):
        outputs[i, i % out_classes] = 1.0
    weights = np.zeros((n_features, out_classes), dtype=float)
    for i in range(n_features):
        for j in range(out_classes):
            weights[i, j] = 1/((i*out_classes)+j+1)
    # print("Weights")
    # print(weights)
    for epoch in range(epochs):
        accuracy = 0
        numCorrect = 0
        logLoss = 0
        for i in range(0, test_size, BATCH_SIZE):
            product = np.dot(inputs[i:i+BATCH_SIZE], weights)
            predicted = softmax(product, axis=1)
            numCorrect += computeAccuracy(predicted, outputs[i:i+BATCH_SIZE])
            # print("Probabilities")
            # print(predicted)
            logLoss += np.sum(-np.multiply(np.log(predicted), outputs[i:i+BATCH_SIZE]))
            grads = predicted - outputs[i:i+BATCH_SIZE]
            updated = np.dot(inputs[i:i+BATCH_SIZE].T, grads)
            weights = weights - (updated * learning_rate / BATCH_SIZE)
            # print("Weights")
            # print(weights)
        accuracy = numCorrect / test_size * 100
        print(f"Accuracy {accuracy}%")
        print(f"Log loss {logLoss}")