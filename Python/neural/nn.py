import numpy as np
from .utils import softmax, sigmoid
import math

class NeuralNetwork:
    def __init__(self, layers, bias=None, weights=None, regularizer=0, learning_rate=0.05):
        self.layers = layers
        self.activations = []
        self.z = []
        self.regularizer = regularizer
        self.learning_rate = learning_rate
        if(weights):
            self.weights = weights
        else: #randomly initialize the weights for each layer
            pass
        pass

    def forward_prop(self, X, verbose=False): #return expected output
        self.activations = []
        self.z = []
        a = np.array(X)
        column = np.ones((len(X), 1))
        a = np.append(column, a, 1)
        if(verbose):
            print("a1: {}\n".format(a[0]))
        self.activations.append(a)
        for i in range(1, len(self.layers)-1):
            a = np.dot(a, np.transpose(self.weights[i-1]))
            if(verbose):
                print("z{}: {}".format(i+1, a[0]))
            self.z.append(a)
            a = sigmoid(a)
            column = np.ones((len(a), 1))
            a = np.append(column, a, 1)
            if(verbose):
                print("a{}: {}\n".format(i+1, a[0]))
            self.activations.append(a)
        a = np.dot(a, np.transpose(self.weights[i]))
        if(verbose):
            print("z{}: {}".format(i+2, a[0]))
        a = sigmoid(a)
        if(verbose):
            print("a{}: {}".format(i+2, a[0]))
        return a

    def cost(self, actual, expected, regularize=True):
        J = 0
        for i in range(len(actual)):
            cost = 0
            for j in range(len(actual[i])):
                cost += ((-expected[i][j]) * (math.log(actual[i][j])) - ((1 - expected[i][j]) * math.log(1 - actual[i][j])))
            J += cost
        pass
        J = J / len(actual)
        if(regularize):
            S = 0
            for i in range(len(self.weights)):
                for j in range(len(self.weights[i])):
                    for k in range(1, len(self.weights[i][j])):
                        S += (self.weights[i][j][k]**2)
            S = (self.regularizer / (2 * len(actual))) * S
            J += S
        return J

    def backprop(self, actual, expected, regularize=True):
        self.gradients = []
        #delta value of the last layer
        deltas = []
        delta3 = actual - expected
        #cost with respect to the weights
        deltas.append(delta3)
        #add an extra zero column to the 
        for i in range(len(self.weights)-1, 0, -1):
            delta2 = np.dot(deltas[0], self.weights[i])
            delta2 = np.multiply(delta2, np.multiply(self.activations[i], (1 - self.activations[i])))
            delta2 = np.delete(delta2, 0, axis=1) #delete the first column
            deltas.insert(0, delta2)
            gradient = np.dot(np.transpose(deltas[i]), self.activations[i])
            if(regularize):
                regularizer = np.multiply(self.regularizer, self.weights[i])
                mask = np.ones(regularizer.shape)
                mask[:,0] = 0
                regularizer = np.multiply(regularizer, mask)
                #multiply the regularizer matrix by a [[0, 1, 1...]...] matrix to not apply the regularizer to the biases
                gradient = (gradient + regularizer) / len(actual)
            else:
                gradient = gradient / len(actual)
            self.gradients.insert(0, gradient)
        pass

    def update_weights(self):
        for i in range(len(self.weights)):
            self.weights[i] = self.weights[i] - np.multiply(self.learning_rate, self.gradients[i])