import numpy as np
from .utils import softMax, sigmoid, relu, get_derivative_for_activation_fn
import math

class NeuralNetwork:
    def __init__(self, layers, bias=None, weights=None, regularizer=0, learning_rate=0.01, activation_fn=relu, final_activation=sigmoid):
        self.layers = layers
        self.activations = []
        self.bias = bias
        self.z = []
        self.regularizer = regularizer
        self.learning_rate = learning_rate
        self.activation_fn = activation_fn
        self.final_activations = final_activation
        if(weights):
            self.weights = weights
        else: #randomly initialize the weights for each layer
            pass
        pass

    def forward_prop(self, X, verbose=False): #return expected output
        self.activations = {}
        self.z = {}
        a = np.array(X)
        if(verbose):
            print("a1: {}\n".format(a))
        self.activations["a0"] = a
        i = 1
        while(i < len(self.layers)-1):
            a = np.dot(a, self.weights[i-1]) + self.bias[i-1]
            self.z[f'z{i}'] = a
            if(verbose):
                print("z{}: {}".format(i+1, a))
            a = self.activation_fn(a)
            if(verbose):
                print("a{}: {}\n".format(i+1, a))
            self.activations[f'a{i}'] = a
            i += 1
        a = np.dot(a, self.weights[i-1]) + self.bias[i-1]
        self.z[f'z{i}'] = a
        if(verbose):
            print("z{}: {}".format(i+2, a[0]))
        a = self.final_activations(a)
        self.activations[f"a{i}"] = a
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
        self.gradients = {}
        self.grad_biases = {}
        #delta value of the last layer
        self.deltas = []
        L = len(self.layers)-1
        #gradient w.r.t. output layer
        delta3 = np.array(actual - expected)
        self.deltas.append(delta3)
        gradient = np.dot(self.activations[f'a{L-1}'].T, delta3)
        grad_bias = np.sum(delta3, axis=0, keepdims=True)
        self.gradients[f'dW{L}'] = gradient
        self.grad_biases[f'db{L}'] = grad_bias

        for i in range(L-1, 0, -1):
            delta2 = np.dot(self.deltas[0], np.transpose(self.weights[i]))
            delta2 = delta2 * get_derivative_for_activation_fn(self.activation_fn)(self.z[f'z{i}'])
            gradient = np.dot(self.activations[f'a{i-1}'].T, delta2)
            grad_bias = np.sum(delta2, axis=0, keepdims=True)
            self.deltas.insert(0, delta2)
            self.gradients[f'dW{i}'] = gradient
            self.grad_biases[f'db{i}'] = grad_bias

        for i in range(L, 0, -1):
            if(regularize):
                mask = self.weights[i-1] * self.regularizer
                self.gradients[f'dW{i}'] += mask
            self.gradients[f'dW{i}'] /= len(actual)
            self.grad_biases[f'db{i}'] /= len(actual)
            
    def update_weights(self):
        for i in range(len(self.weights)):
            self.weights[i] -= self.learning_rate * self.gradients[f'dW{i+1}']
            self.bias[i] -= self.learning_rate * self.grad_biases[f'db{i+1}']