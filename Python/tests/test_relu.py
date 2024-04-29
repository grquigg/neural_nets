import unittest
import numpy as np
from context import NeuralNetwork, softmax, sigmoid, relu, cross_entropy_loss

class TestRelu(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.layers = [2,2,2]
        cls.bias = [np.array([[0.0, 0.0]]),np.array([[0.0,0.0]])]
        cls.weights = [np.array([[0.1, 0.2],[0.3,0.4]]), np.array([[0.5,0.6],[0.7,0.8]])]
        cls.model = NeuralNetwork(cls.layers, bias=cls.bias, weights=cls.weights, activation_fn=relu, final_activation=softmax, learning_rate=0.001)
        cls.x = [[0,1],[1,0]]
        cls.y = [[1,0], [0,1]]

    def test_forward_activation(self):
        self.model.forward_prop(self.x)
        self.assertTrue(np.allclose(self.model.activations[0], self.x))
        self.assertTrue(np.allclose(self.model.z[0], [[0.3,0.4],[0.1,0.2]]))
        self.assertTrue(np.allclose(self.model.activations[1], [[0.3,0.4],[0.1,0.2]]))
        self.assertTrue(np.allclose(self.model.z[1], [[0.43,0.5],[0.19,0.22]]))
        self.assertTrue(np.allclose(self.model.activations[2], [[0.48250714,0.51749286],[0.49250056,0.50749944]]))
        self.model.backprop(self.y, self.model.activations[-1], regularize=False)
        self.assertEqual(self.model.deltas[-1].shape, (2,2))
        self.assertTrue(np.allclose(self.model.deltas[-1], [[0.5175,-0.5175],[-0.4925,0.4925]], atol=1e-4))
        self.assertTrue(np.allclose(self.model.gradients[-1], [[0.053, -0.053], [0.054, -0.054]], atol=1e-3))
        self.assertTrue(np.allclose(self.model.deltas[-2], [[-0.05175,-0.05175],[0.04925,0.04925]], atol=1e-3))
        self.assertTrue(np.allclose(self.model.gradients[-2], [[0.0246,0.0246],[-0.0258,-0.0258]], atol=1e-3))
    
    def test_training(self):
        self.model.activation_fn = sigmoid
        self.model.forward_prop(self.x)
        self.model.backprop(self.y, self.model.activations[-1], regularize=False)
        self.model.update_weights()
        self.assertEqual(self.model.weights[0].shape, (2,2))
        print("New weights")
        print(self.model.weights[0])
    
    def test_simple_relu(self):
        pass
if __name__ == "__main__":
    unittest.main()