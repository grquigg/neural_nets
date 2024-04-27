import unittest
import numpy as np
from context import NeuralNetwork, softmax, sigmoid, relu

class TestNNExampleOne(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.layers = [1,2,1]
        cls.bias = [np.array([[0.4, 0.3]]),np.array([[0.7]])]
        cls.weights = [np.array([[0.1, 0.2]]), np.array([[0.5],[0.6]])]
        cls.model = NeuralNetwork(cls.layers, bias=cls.bias, weights=cls.weights, activation_fn=sigmoid)
        cls.x = [[0.13],[0.42]]
        cls.y = [[0.9], [0.23]]

    @classmethod
    def tearDownModule(cls):
        pass

    def test_activations_for_ex_one(self):
        correct = [[0.13],[0.60181,0.58079],[0.79403]]
        self.model.forward_prop(self.x[0])
        self.assertEqual(self.model.activations[0], correct[0])
        for i in range(1, len(self.model.activations)):
            for j in range(len(self.model.activations[i])):
                self.assertAlmostEqual(self.model.activations[i][0][j], correct[i][j], places=5)

    def test_activations_for_ex_two(self):
        correct = [[0.42],[0.60874,0.59484],[0.79597]]
        self.model.forward_prop(self.x[1])
        self.assertEqual(self.model.activations[0], correct[0])
        for i in range(1, len(self.model.activations)):
            for j in range(len(self.model.activations[i])):
                self.assertAlmostEqual(self.model.activations[i][0][j], correct[i][j], places=5)

    def test_final_activation_softmax_for_ex_one(self):
        self.model.final_activations = softmax
        self.model.forward_prop(self.x[0])
        self.assertEqual(self.model.activations[2], 1.0)

    def test_final_activation_softmax_for_ex_two(self):
        self.model.final_activations = softmax
        self.model.forward_prop(self.x[1])
        self.assertEqual(self.model.activations[2], 1.0)

    def test_activations_for_batch_size_two(self):
        self.model.forward_prop(self.x)
        self.assertTrue(np.allclose(self.model.activations[0], [[0.13],[0.42]]))
        self.assertTrue(np.allclose(self.model.activations[1], [[0.60181,0.58079],[0.60874,0.59484]]))
        self.assertTrue(np.allclose(self.model.activations[2], [[0.79403],[0.79597]]))

    def test_deltas_for_ex_one(self):
        self.model.activation_fn = sigmoid
        self.model.forward_prop(self.x[0])
        self.model.backprop(self.model.activations[-1], self.y[0], regularize=False)
        self.assertTrue(np.allclose(self.model.deltas[1], [[-0.10597]], rtol=1e-4))
        self.assertEqual(self.model.deltas[1].shape, (1,1))
        self.assertEqual(self.model.deltas[0].shape, (1,2))
        self.assertTrue(np.allclose(self.model.deltas[0], [[-0.01270, -0.01548]], rtol=1e-3))
        self.assertEqual(self.model.gradients[1].shape, self.model.weights[1].shape)
        self.assertTrue(np.allclose(self.model.gradients[1], [[-0.06378],[-0.06155]], rtol=1e-3))
        self.assertTrue(np.allclose(self.model.gradients[0], [[-0.00165, -0.00201]], atol=1e-3))

    def test_deltas_for_ex_two(self):
        self.model.activation_fn = sigmoid
        self.model.forward_prop(self.x[1])
        self.model.backprop(self.model.activations[-1], self.y[1], regularize=False)
        self.assertTrue(np.allclose(self.model.deltas[1], [[0.56597]]))
        self.assertTrue(np.allclose(self.model.deltas[0], [[0.06740,0.08184]], rtol=1e-3))
        self.assertTrue(np.allclose(self.model.gradients[1], [[0.34452],[0.33666]], atol=1e-3))
        self.assertTrue(np.allclose(self.model.gradients[0], [[0.02831, 0.03437]], atol=1e-3))
    
    def test_deltas_for_both(self):
        self.model.activation_fn = sigmoid
        self.model.forward_prop(self.x)
        self.model.backprop(self.model.activations[-1], self.y, regularize=False)
        self.assertTrue(np.allclose(self.model.gradients[1], [[0.14037],[0.13756]], atol=1e-5))
        self.assertTrue(np.allclose(self.model.gradients[0], [[0.01333, 0.01618]], atol=1e-5))
        self.assertTrue(np.allclose(self.model.grad_biases[0], [0.02735, 0.03318], atol=1e-5))
        self.assertTrue(np.allclose(self.model.grad_biases[1], [0.23], atol=1e-5))

    def test_forward_prop_for_relu_ex_one(self):
        self.model.final_activations = sigmoid
        self.model.activation_fn = relu
        self.model.forward_prop(self.x[0])
        self.assertTrue(np.allclose(self.model.activations[0], [0.13], atol=1e-5))
        self.assertTrue(np.allclose(self.model.activations[1], [[0.413, 0.326]], atol=1e-5))
        self.assertTrue(np.allclose(self.model.activations[2], [[0.75065]], atol=1e-5))

    def test_forward_prop_for_relu_ex_two(self):
        self.model.final_activations = sigmoid
        self.model.activation_fn = relu
        self.model.forward_prop(self.x[1])
        self.assertTrue(np.allclose(self.model.activations[0], [0.42], atol=1e-5))
        self.assertTrue(np.allclose(self.model.activations[1], [[0.442, 0.384]], atol=1e-5))
        self.assertTrue(np.allclose(self.model.activations[2], [[0.75976654]], atol=1e-5))

    def test_backprop_for_relu_ex_one(self):
        self.model.activation_fn = relu
        self.model.forward_prop(self.x[0])
        self.model.backprop(self.model.activations[-1], self.y[0], regularize=False)
        print(self.model.deltas[0])
        self.assertTrue(np.allclose(self.model.deltas[1], [[0.75065-0.9]], rtol=1e-4))
        self.assertTrue(np.allclose(self.model.deltas[0]),[[]])

if __name__ == "__main__":
    unittest.main()