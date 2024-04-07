import unittest
import numpy as np
from context import NeuralNetwork, softmax

class TestNNExampleOne(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.layers = [1,2,1]
        cls.bias = [[[0.4, 0.3]],[[0.7]]]
        cls.weights = [[[0.1, 0.2]], [[0.5],[0.6]]]
        cls.model = NeuralNetwork(cls.layers, bias=cls.bias, weights=cls.weights)
        cls.x = [[0.13],[0.42]]
        cls.y = [[0.9], [0.23]]

    @classmethod
    def tearDownModule(cls):
        pass

    def test_activations_for_ex_one(self):
        correct = [[0.13],[0.60181,0.58079],[0.79403]]
        self.model.forward_prop(self.x[0], verbose=True)
        self.assertEqual(self.model.activations[0], correct[0])
        for i in range(1, len(self.model.activations)):
            for j in range(len(self.model.activations[i])):
                self.assertAlmostEqual(self.model.activations[i][0][j], correct[i][j], places=5)

    def test_activations_for_ex_two(self):
        correct = [[0.42],[0.60874,0.59484],[0.79597]]
        self.model.forward_prop(self.x[1], verbose=True)
        self.assertEqual(self.model.activations[0], correct[0])
        for i in range(1, len(self.model.activations)):
            for j in range(len(self.model.activations[i])):
                self.assertAlmostEqual(self.model.activations[i][0][j], correct[i][j], places=5)

    def test_final_activation_softmax_for_ex_one(self):
        self.model.final_activations = softmax
        self.model.forward_prop(self.x[0])
        print(self.model.activations)
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

        
if __name__ == "__main__":
    unittest.main()