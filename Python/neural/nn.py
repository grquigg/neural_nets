import numpy as np
from .utils import softmax, sigmoid
import math

class NeuralNetwork:
    def __init__(self, layers, bias=None, weights=None, regularizer=0, learning_rate=0.05, final_activation=sigmoid):
        self.layers = layers
        self.activations = []
        self.bias = bias
        self.z = []
        self.regularizer = regularizer
        self.learning_rate = learning_rate
        self.final_activations = final_activation
        if(weights):
            self.weights = weights
        else: #randomly initialize the weights for each layer
            pass
        pass

    def forward_prop(self, X, verbose=False): #return expected output
        self.activations = []
        self.z = []
        a = np.array(X)
        if(verbose):
            print("a1: {}\n".format(a))
        self.activations.append(a)
        i = 1
        while(i < len(self.layers)-1):
            a = np.dot(a, self.weights[i-1]) + self.bias[i-1]
            if(verbose):
                print("z{}: {}".format(i+1, a))
            self.z.append(a)
            a = sigmoid(a)
            if(verbose):
                print("a{}: {}\n".format(i+1, a))
            self.activations.append(a)
            i += 1
        a = np.dot(a, self.weights[i-1]) + self.bias[i-1]
        if(verbose):
            print("z{}: {}".format(i+2, a[0]))
        a = self.final_activations(a)
        self.activations.append(a)
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
        self.grad_biases = []
        #delta value of the last layer
        self.deltas = []
        delta3 = np.array(actual - expected)
        #cost with respect to the weights
        #expect delta3 to be of size [batch_size, output_size]
        
        self.deltas.append(delta3)
        #add an extra zero column to the 
        for i in range(len(self.weights)-1, 0, -1):
            delta2 = np.dot(self.deltas[0], np.transpose(self.weights[i]))
            delta2 = np.multiply(delta2, np.multiply(self.activations[i], (1 - self.activations[i])))
            self.deltas.insert(0, delta2)

        for i in range(len(self.weights)-1, -1, -1):
            gradient = np.dot(np.transpose(self.activations[i]),self.deltas[i])
            grad_bias = np.sum(self.deltas[i], axis=0) / len(actual)
            if(regularize):
                regularizer = np.multiply(self.regularizer, self.weights[i])
                mask = np.ones(regularizer.shape)
                regularizer = np.multiply(regularizer, mask)
                #multiply the regularizer matrix by a [[0, 1, 1...]...] matrix to not apply the regularizer to the biases
                gradient = (gradient + regularizer) / len(actual)

                regularized = np.multiply(self.regularizer, self.bias[i])
                mask = np.ones(regularized.shape)
                regularized = np.multiply(regularized, mask)
                
            else:
                gradient = gradient / len(actual)
            self.gradients.insert(0, gradient)
            self.grad_biases.insert(0, grad_bias)

    def update_weights(self):
        for i in range(len(self.weights)):
            self.weights[i] = self.weights[i] - np.multiply(self.learning_rate, self.gradients[i])
            self.bias[i] = self.bias[i] - np.multiply(self.learning_rate, self.grad_biases[i])