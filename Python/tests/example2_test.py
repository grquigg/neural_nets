import unittest
import numpy as np
from context import NeuralNetwork, softmax

class TestNNExampleOne(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.layers = [2,4,3,2]
        cls.weights = [np.array([[0.15, 0.1, 0.19, 0.35], [0.4, 0.54, 0.42, 0.68]]),np.array([[0.67, 0.42, 0.56], [0.14, 0.2, 0.8], [0.96, 0.32, 0.69], [0.87, 0.89, 0.09]]), np.array([[0.87, 0.1], [0.42, 0.95], [0.53, 0.69]])]
        cls.bias = [np.array([[0.42, 0.72, 0.01, 0.3]]), np.array([[0.21, 0.87, 0.03]]), np.array([[0.04, 0.17]])]
        cls.model = NeuralNetwork(cls.layers, bias=cls.bias, weights=cls.weights)
        cls.x = [[0.32, 0.68],[0.83, 0.02]]
        cls.y = [[0.75, 0.98], [0.75, 0.28]]

    @classmethod
    def tearDownModule(cls):
        pass

    def test_activations_for_ex_one(self):
        correct = [[0.32000,0.68000],[0.67700,0.75384,0.58817,0.70566],[0.87519,0.89296,0.81480],[0.83318, 0.84132]]
        self.model.forward_prop(self.x[0])
        for i in range(len(self.model.activations)):
            self.assertTrue(np.allclose(correct[i], self.model.activations[i], atol=1e-5))

    def test_activations_for_ex_two(self):
        correct = [[0.83000,0.02000],[0.63472,0.69292,0.54391,0.64659],[0.86020,0.88336,0.79791],[0.82953,0.83832]]
        self.model.forward_prop(self.x[1])
        for i in range(len(self.model.activations)):
            self.assertTrue(np.allclose(correct[i], self.model.activations[i], atol=1e-5))

    def test_final_activation_softmax_for_ex_one(self):
        self.model.final_activations = softmax
        self.model.forward_prop(self.x[0])
        
        self.assertTrue(np.allclose(self.model.activations[-1], [[0.4850698, 0.5149302]]))

    def test_final_activation_softmax_for_ex_two(self):
        self.model.final_activations = softmax
        self.model.forward_prop(self.x[1])
        self.assertTrue(np.allclose(self.model.activations[-1], [[0.4841319, 0.5158681]]))

    def test_activations_for_batch_size_two(self):
        self.model.forward_prop(self.x)
        self.assertTrue(np.allclose(self.model.activations[0], [[0.32000,0.68000],[0.83000,0.02000]]))
        self.assertTrue(np.allclose(self.model.activations[1], [[0.67700,0.75384,0.58817,0.70566],[0.63472,0.69292,0.54391,0.64659]]))
        self.assertTrue(np.allclose(self.model.activations[2], [[0.87519,0.89296,0.81480],[0.86020,0.88336,0.79791]]))
        self.assertTrue(np.allclose(self.model.activations[3], [[0.83318, 0.84132], [0.82953,0.83832]]))

    def test_deltas_for_ex_one(self):
        self.model.forward_prop([self.x[0]])
        self.model.backprop(self.model.activations[-1], self.y[0], regularize=False)
        self.assertTrue(np.allclose(self.model.deltas[2], [[0.08318,-0.13868]], atol=1e-5))
        self.assertTrue(np.allclose(self.model.deltas[1], [[0.00639,-0.00925,-0.00779]], atol=1e-5))
        self.assertTrue(np.allclose(self.model.deltas[0], [[-0.00087,-0.00133,-0.00053,-0.00070]], atol=1e-5))
        self.assertEqual(self.model.deltas[2].shape, (1,2))
        self.assertEqual(self.model.deltas[1].shape, (1,3))
        self.assertEqual(self.model.deltas[0].shape, (1,4))
        for i in range(len(self.model.gradients)):
            self.assertEqual(self.model.gradients[i].shape, self.model.weights[i].shape)
        self.assertTrue(np.allclose(self.model.gradients[2], [[0.07280,-0.12138],[0.07427,-0.12384],[0.0677,-0.113]], atol=1e-4))
        self.assertTrue(np.allclose(self.model.gradients[1], [[0.00433, -0.00626, -0.00527], [0.00482, -0.00698, -0.00587], [0.00376, -0.00544, -0.00458], [0.00451, -0.00653, -0.00550]], atol=1e-4))
        self.assertTrue(np.allclose(self.model.gradients[0], [[-0.00028, -0.00043, -0.00017, -0.00022], [-0.00059, -0.00091, -0.00036, -0.00048]], atol=1e-4))
        self.assertTrue(np.allclose(self.model.grad_biases[0], [-0.00087,-0.00133,-0.00053,-0.00070], atol=1e-5))
        self.assertTrue(np.allclose(self.model.grad_biases[1], [0.00639,-0.00925,-0.00779], atol=1e-5))
        self.assertTrue(np.allclose(self.model.grad_biases[2], [0.08318,-0.13868], atol=1e-5))
    
    def test_deltas_for_ex_two(self):
        self.model.forward_prop([self.x[1]])
        self.model.backprop(self.model.activations[-1], self.y[1], regularize=False)
        self.assertTrue(np.allclose(self.model.deltas[2], [[0.07953,0.55832]], atol=1e-5))
        self.assertTrue(np.allclose(self.model.deltas[1], [[0.01503,0.05809,0.06892]], atol=1e-5))
        self.assertTrue(np.allclose(self.model.deltas[0], [[0.01694,0.01465,0.01999,0.01622]], atol=1e-5))
        self.assertEqual(self.model.deltas[2].shape, (1,2))
        self.assertEqual(self.model.deltas[1].shape, (1,3))
        self.assertEqual(self.model.deltas[0].shape, (1,4))
        for i in range(len(self.model.gradients)):
            self.assertEqual(self.model.gradients[i].shape, self.model.weights[i].shape)
        self.assertTrue(np.allclose(self.model.gradients[2], [[0.06841,0.48027],[0.07025,0.49320],[0.06346,0.44549]], atol=1e-4))
        self.assertTrue(np.allclose(self.model.gradients[1], [[0.00954, 0.03687, 0.04374], [0.01042, 0.04025, 0.04775], [0.00818, 0.03160, 0.03748], [0.00972, 0.03756, 0.04456]], atol=1e-4))
        self.assertTrue(np.allclose(self.model.gradients[0], [[0.01406, 0.01216, 0.01659, 0.01346], [0.00034, 0.00029, 0.00040, 0.00032]], atol=1e-4))

    def test_deltas_for_both(self):
        self.model.forward_prop(self.x)
        self.model.regularizer = 0.250
        self.model.backprop(self.model.activations[-1], self.y, regularize=True)
        self.assertTrue(np.allclose(self.model.gradients[0], [[0.02564, 0.01837, 0.03196, 0.05037],[0.04987, 0.06719, 0.05252, 0.08492]], atol=1e-4))
        self.assertTrue(np.allclose(self.model.gradients[1], [[0.09068, 0.06780, 0.08924], [0.02512, 0.04164, 0.12094], [0.12597, 0.05308, 0.10270], [0.11586, 0.12677, 0.03078]], atol=1e-4))
        self.assertTrue(np.allclose(self.model.gradients[2], [[0.17935, 0.19195], [0.12476, 0.30343], [0.13186, 0.25249]], atol=1e-4))
        self.assertTrue(np.allclose(self.model.grad_biases[0], [0.00804, 0.00666, 0.00973, 0.00776], atol=1e-4))
        self.assertTrue(np.allclose(self.model.grad_biases[1], [0.01071, 0.02442, 0.03056], atol=1e-4))
        self.assertTrue(np.allclose(self.model.grad_biases[2], [0.08135, 0.20982], atol=1e-4))
    
if __name__ == "__main__":
    unittest.main()