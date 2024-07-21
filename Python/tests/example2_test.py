import unittest
import numpy as np
from context import NeuralNetwork, softMax, sigmoid, relu

class TestNNExampleOne(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.layers = [2,4,3,2]
        cls.weights = [np.array([[0.15, 0.1, 0.19, 0.35], [0.4, 0.54, 0.42, 0.68]]),np.array([[0.67, 0.42, 0.56], [0.14, 0.2, 0.8], [0.96, 0.32, 0.69], [0.87, 0.89, 0.09]]), np.array([[0.87, 0.1], [0.42, 0.95], [0.53, 0.69]])]
        cls.bias = [np.array([[0.42, 0.72, 0.01, 0.3]]), np.array([[0.21, 0.87, 0.03]]), np.array([[0.04, 0.17]])]
        cls.model = NeuralNetwork(cls.layers, bias=cls.bias, weights=cls.weights, activation_fn=sigmoid)
        cls.x = [[0.32, 0.68],[0.83, 0.02]]
        cls.y = [[0.75, 0.98], [0.75, 0.28]]

    @classmethod
    def tearDownModule(cls):
        cls.model.activation_fn = sigmoid

    def test_activations_for_ex_one(self):
        self.model.final_activations = sigmoid
        correct = [[0.32000,0.68000],[0.67700,0.75384,0.58817,0.70566],[0.87519,0.89296,0.81480],[0.83318, 0.84132]]
        self.model.forward_prop(self.x[0])
        for i in range(len(self.model.activations)):
            self.assertTrue(np.allclose(correct[i], self.model.activations[f'a{i}'], atol=1e-5))

    def test_activations_for_ex_two(self):
        correct = [[0.83000,0.02000],[0.63472,0.69292,0.54391,0.64659],[0.86020,0.88336,0.79791],[0.82953,0.83832]]
        self.model.forward_prop(self.x[1])
        for i in range(len(self.model.activations)):
            self.assertTrue(np.allclose(correct[i], self.model.activations[f'a{i}'], atol=1e-5))

    def test_final_activation_softMax_for_ex_one(self):
        self.model.activation_fn = sigmoid
        self.model.final_activations = softMax
        output = self.model.forward_prop(self.x[0])
        
        self.assertTrue(np.allclose(output, [[0.4850698, 0.5149302]]))

    def test_final_activation_softMax_for_ex_two(self):
        self.model.activation_fn = sigmoid
        self.model.final_activations = softMax
        output = self.model.forward_prop(self.x[1])
        self.assertTrue(np.allclose(output, [[0.4841319, 0.5158681]]))

    def test_activations_for_batch_size_two(self):
        self.model.forward_prop(self.x)
        self.assertTrue(np.allclose(self.model.activations['a0'], [[0.32000,0.68000],[0.83000,0.02000]]))
        self.assertTrue(np.allclose(self.model.activations['a1'], [[0.67700,0.75384,0.58817,0.70566],[0.63472,0.69292,0.54391,0.64659]]))
        self.assertTrue(np.allclose(self.model.activations['a2'], [[0.87519,0.89296,0.81480],[0.86020,0.88336,0.79791]]))
        self.assertTrue(np.allclose(self.model.activations['a3'], [[0.83318, 0.84132], [0.82953,0.83832]]))

    def test_deltas_for_ex_one(self):
        self.model.activation_fn = sigmoid
        self.model.final_activations = sigmoid
        out = self.model.forward_prop([self.x[0]])
        self.model.backprop(out, self.y[0], regularize=False)
        self.assertTrue(np.allclose(self.model.deltas[2], [[0.08318,-0.13868]], atol=1e-5))
        self.assertTrue(np.allclose(self.model.deltas[1], [[0.00639,-0.00925,-0.00779]], atol=1e-5))
        self.assertTrue(np.allclose(self.model.deltas[0], [[-0.00087,-0.00133,-0.00053,-0.00070]], atol=1e-5))
        self.assertEqual(self.model.deltas[2].shape, (1,2))
        self.assertEqual(self.model.deltas[1].shape, (1,3))
        self.assertEqual(self.model.deltas[0].shape, (1,4))
        for i in range(len(self.model.gradients)):
            self.assertEqual(self.model.gradients[f'dW{i+1}'].shape, self.model.weights[i].shape)
        self.assertTrue(np.allclose(self.model.gradients['dW3'], [[0.07280,-0.12138],[0.07427,-0.12384],[0.0677,-0.113]], atol=1e-4))
        self.assertTrue(np.allclose(self.model.gradients['dW2'], [[0.00433, -0.00626, -0.00527], [0.00482, -0.00698, -0.00587], [0.00376, -0.00544, -0.00458], [0.00451, -0.00653, -0.00550]], atol=1e-4))
        self.assertTrue(np.allclose(self.model.gradients['dW1'], [[-0.00028, -0.00043, -0.00017, -0.00022], [-0.00059, -0.00091, -0.00036, -0.00048]], atol=1e-4))
        self.assertTrue(np.allclose(self.model.grad_biases['db1'], [-0.00087,-0.00133,-0.00053,-0.00070], atol=1e-5))
        self.assertTrue(np.allclose(self.model.grad_biases['db2'], [0.00639,-0.00925,-0.00779], atol=1e-5))
        self.assertTrue(np.allclose(self.model.grad_biases['db3'], [0.08318,-0.13868], atol=1e-5))
    
    def test_deltas_for_ex_two(self):
        self.model.activation_fn = sigmoid
        self.model.final_activations = sigmoid
        out = self.model.forward_prop([self.x[1]])
        self.model.backprop(out, self.y[1], regularize=False)
        self.assertTrue(np.allclose(self.model.deltas[2], [[0.07953,0.55832]], atol=1e-5))
        self.assertTrue(np.allclose(self.model.deltas[1], [[0.01503,0.05809,0.06892]], atol=1e-5))
        self.assertTrue(np.allclose(self.model.deltas[0], [[0.01694,0.01465,0.01999,0.01622]], atol=1e-5))
        self.assertEqual(self.model.deltas[2].shape, (1,2))
        self.assertEqual(self.model.deltas[1].shape, (1,3))
        self.assertEqual(self.model.deltas[0].shape, (1,4))
        for i in range(len(self.model.gradients)):
            self.assertEqual(self.model.gradients[f'dW{i+1}'].shape, self.model.weights[i].shape)
        self.assertTrue(np.allclose(self.model.gradients['dW3'], [[0.06841,0.48027],[0.07025,0.49320],[0.06346,0.44549]], atol=1e-4))
        self.assertTrue(np.allclose(self.model.gradients['dW2'], [[0.00954, 0.03687, 0.04374], [0.01042, 0.04025, 0.04775], [0.00818, 0.03160, 0.03748], [0.00972, 0.03756, 0.04456]], atol=1e-4))
        self.assertTrue(np.allclose(self.model.gradients['dW1'], [[0.01406, 0.01216, 0.01659, 0.01346], [0.00034, 0.00029, 0.00040, 0.00032]], atol=1e-4))

    """
    Also a good check to see whether or not the implementation for regularization actually works or not.
    """
    def test_deltas_for_both(self):
        self.model.activation_fn = sigmoid
        self.model.final_activations = sigmoid
        out = self.model.forward_prop(self.x)
        self.model.regularizer = 0.250
        self.model.backprop(out, self.y, regularize=True)
        self.assertTrue(np.allclose(self.model.gradients['dW1'], [[0.02564, 0.01837, 0.03196, 0.05037],[0.04987, 0.06719, 0.05252, 0.08492]], atol=1e-4))
        self.assertTrue(np.allclose(self.model.gradients['dW2'], [[0.09068, 0.06780, 0.08924], [0.02512, 0.04164, 0.12094], [0.12597, 0.05308, 0.10270], [0.11586, 0.12677, 0.03078]], atol=1e-4))
        self.assertTrue(np.allclose(self.model.gradients['dW3'], [[0.17935, 0.19195], [0.12476, 0.30343], [0.13186, 0.25249]], atol=1e-4))
        self.assertEqual(self.model.grad_biases['db1'].shape, (1,4))
        self.assertTrue(np.allclose(self.model.grad_biases['db1'], [0.00804, 0.00666, 0.00973, 0.00776], atol=1e-4))
        self.assertTrue(np.allclose(self.model.grad_biases['db2'], [0.01071, 0.02442, 0.03056], atol=1e-4))
        self.assertTrue(np.allclose(self.model.grad_biases['db3'], [0.08135, 0.20982], atol=1e-4))

    #we should test that the forward pass is working as we might expect it to be
    def test_forward_pass_relu(self):
        self.model.activation_fn = relu
        self.model.final_activations = softMax
        self.model.forward_prop(self.x)
        self.assertTrue(np.allclose(self.model.activations['a1'], [[0.74, 1.1192, 0.3564, 0.8744], [0.5525, 0.8138, 0.1761, 0.6041]], atol=1e-4))
        self.assertTrue(np.allclose(self.model.activations['a2'], [[1.96536 , 2.296904, 1.664372], [1.38873 , 1.858811, 1.166318]], atol=1e-4))
        self.assertTrue(np.allclose(self.model.activations['a3'], [[0.47493816, 0.52506184], [0.44214564, 0.55785436]]))
        self.assertTrue(np.allclose(np.sum(self.model.activations['a3'], axis=1), [[1.0, 1.0]]))
    
    #important note that the deltas should be unaffected by whether or not we actually include regularization as a parameter in the network
    def test_backward_pass_relu(self):
        self.model.activation_fn = relu
        self.model.final_activations = softMax
        self.model.regularizer = 0.250
        out = self.model.forward_prop(self.x)
        self.model.backprop(out, self.y, regularize=True)
        self.assertTrue(np.allclose(self.model.deltas[2], [[-0.27506184, -0.45493816],[-0.30785436,  0.27785436]], atol=1e-5))
        self.assertTrue(np.allclose(self.model.deltas[1], [[-0.28479762, -0.54771722, -0.45969011], [-0.24004786,  0.13466281,  0.0285567]]))
        self.assertTrue(np.allclose(self.model.deltas[0], [[-0.6782821 , -0.5171672 , -0.7658614 , -0.77661437], [-0.08828193,  0.01617122, -0.16764972, -0.08642163]], atol=1e-5))
        self.assertTrue(np.allclose(self.model.gradients['dW3'], [[-0.37531106, -0.24162629],
       [-0.54951686, -0.14548527],
       [-0.34218066, -0.13030989]]))
        self.assertTrue(np.allclose(self.model.gradients['dW2'], [[-0.08793834, -0.11295477, -0.09219655],
       [-0.23954822, -0.22670826, -0.14562286],
       [ 0.04811285, -0.04574615,  0.00684764],
       [-0.08826997, -0.08753707, -0.18110096]]))
        self.assertTrue(np.allclose(self.model.gradients['dW1'], [[-0.12641214, -0.06353569, -0.16836246, -0.11637328],
       [-0.18149873, -0.10817513, -0.20956937, -0.1799131 ]]))
        self.assertTrue(np.allclose(self.model.grad_biases['db3'], [[-0.2914581, -0.0885419]]))
        self.assertTrue(np.allclose(self.model.grad_biases['db2'], [[-0.26242274, -0.20652721, -0.2155667 ]]))
        self.assertTrue(np.allclose(self.model.grad_biases['db1'], [[-0.38328202, -0.25049799, -0.46675556, -0.431518 ]]))


if __name__ == "__main__":
    unittest.main()