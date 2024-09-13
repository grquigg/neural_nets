classdef ExampleTestCases2 < matlab.unittest.TestCase
    properties
        nLayers = 4;
        layers = [];
        model;
        weights;
        biases;
        activations;
        inputs;
        outputs;
    end

    methods(TestMethodSetup)
        function createFigure(testCase)
            testCase.layers = [2,4,3,2];
            testCase.weights = cell(3,1);
            testCase.weights{1} = [0.15000,0.40000;0.10000,0.54000;0.19000,0.42000;0.35000,0.68000];
            testCase.weights{2} = [0.67000,0.14000,0.96000,0.87000;0.42000,0.20000,0.32000,0.89000;0.56000,0.80000,0.69000,0.09000];
            testCase.weights{3} = [0.87000,0.42000,0.53000; 0.10000,0.9500,0.69000];
            for i=1:size(testCase.layers,2)-1
                testCase.weights{i} = testCase.weights{i}';
            end
            testCase.biases = cell(3,1);
            testCase.biases{1} = [0.42, 0.72, 0.01,0.3];
            testCase.biases{2} = [0.21, 0.87, 0.03];
            testCase.biases{3} = [0.04, 0.17];
            testCase.model = NeuralNetwork(testCase.layers, testCase.weights, testCase.biases);
            testCase.inputs = [0.32,0.68;0.83,0.02];
            testCase.outputs = [0.75000,0.98000;0.75000,0.28000];
            testCase.activations = cell(3,1);
        end

    end

    methods(TestMethodTeardown)
        function closeFigure(testCase)
        end
    end
    
    methods(Test)
        function dotProduct1(testCase)
            % Test to check some default properties of the figure
            testCase.activations{1} = (testCase.inputs * testCase.model.weights{1}) + testCase.model.biases{1};
            expected = [0.74000,1.11920,0.35640,0.87440;0.55250,0.81380,0.17610,0.60410];
            testCase.verifyEqual(testCase.activations{1}, expected, "AbsTol", 1e-5);
        end

        function sigmoid1(testCase)
            testCase.activations{1} = (testCase.inputs * testCase.model.weights{1}) + testCase.model.biases{1};
            testCase.activations{1} = utils.sigmoid(testCase.activations{1});
            expected = [0.67700,0.75384,0.58817,0.70566;0.63472,0.69292,0.54391,0.64659];
            testCase.verifyEqual(testCase.activations{1}, expected, "AbsTol", 1e-5);
        end

        function dotProduct2(testCase)
            testCase.activations{1} = (testCase.inputs * testCase.model.weights{1}) + testCase.model.biases{1};
            testCase.activations{1} = utils.sigmoid(testCase.activations{1});
            testCase.activations{2} = (testCase.activations{1} * testCase.model.weights{2}) + testCase.model.biases{2}; 
            expected = [1.94769,2.12136,1.48154;1.81696,2.02468,1.37327];
            testCase.verifyEqual(testCase.activations{2}, expected, "AbsTol", 1e-5);
        end

        function sigmoidProduct2(testCase)
            testCase.activations{1} = (testCase.inputs * testCase.model.weights{1}) + testCase.model.biases{1};
            testCase.activations{1} = utils.sigmoid(testCase.activations{1});
            testCase.activations{2} = (testCase.activations{1} * testCase.model.weights{2}) + testCase.model.biases{2}; 
            testCase.activations{2} = utils.sigmoid(testCase.activations{2});
            expected = [0.87519,0.89296,0.81480;0.86020,0.88336,0.79791];
            testCase.verifyEqual(testCase.activations{2}, expected, "AbsTol", 1e-5);
        end

        function dotProduct3(testCase)
            testCase.activations{1} = (testCase.inputs * testCase.model.weights{1}) + testCase.model.biases{1};
            testCase.activations{1} = utils.sigmoid(testCase.activations{1});
            testCase.activations{2} = (testCase.activations{1} * testCase.model.weights{2}) + testCase.model.biases{2}; 
            testCase.activations{2} = utils.sigmoid(testCase.activations{2});
            testCase.activations{3} = (testCase.activations{2} * testCase.model.weights{3}) + testCase.model.biases{3}; 
            expected = [1.60831,1.66805;1.58228,1.64577];
            testCase.verifyEqual(testCase.activations{3}, expected, "AbsTol", 1e-5);
        end

        function sigmoidProduct3(testCase)
            testCase.activations{1} = (testCase.inputs * testCase.model.weights{1}) + testCase.model.biases{1};
            testCase.activations{1} = utils.sigmoid(testCase.activations{1});
            testCase.activations{2} = (testCase.activations{1} * testCase.model.weights{2}) + testCase.model.biases{2}; 
            testCase.activations{2} = utils.sigmoid(testCase.activations{2});
            testCase.activations{3} = (testCase.activations{2} * testCase.model.weights{3}) + testCase.model.biases{3}; 
            testCase.activations{3} = utils.sigmoid(testCase.activations{3});  
            expected = [0.83318,0.84132;0.82953,0.83832];
            testCase.verifyEqual(testCase.activations{3}, expected, "AbsTol", 1e-5);
        end

        function forwardPassSoftmaxFinal(testCase)
            testCase.model.final_activation_fn = @utils.softmax;
            testCase.model.forward_pass(testCase.inputs);
            expected = [0.4850698,0.5149;0.484,0.5159];
            testCase.verifyEqual(testCase.model.activations{4}, expected, "AbsTol", 1e-3);
        end

        function forwardPassTestSigmoidFinalLayer(testCase)
            testCase.model.forward_pass(testCase.inputs);
            expected = [0.83318,0.84132; 0.82953,0.83832];
            testCase.verifyEqual(testCase.model.activations{4}, expected, "AbsTol", 1e-5);
        end

        function logLossSoftmax(testCase)
            testCase.model.final_activation_fn = @utils.softmax;
            testCase.model.forward_pass(testCase.inputs);
            loss = utils.crossEntropyLoss(testCase.outputs, testCase.model.activations{4});
            testCase.verifyEqual(loss, 1.922427885698340, "AbsTol", 1e-5);
        end

        function testBackpropDelta(testCase)
            testCase.model.final_activation_fn = @utils.softmax;
            testCase.model.forward_pass(testCase.inputs);
            testCase.model.backprop(testCase.inputs, testCase.outputs, false);
            testCase.verifyEqual(testCase.model.deltas{3}, [-0.2649302,-0.4650698;-0.2658681,0.2358681], "AbsTol", 1e-6);
            testCase.verifyEqual(testCase.model.deltas{2}, [-0.03025601,-0.05286462,-0.06961101;-0.02497924,0.0115818,0.00352154], "AbsTol", 1e-6);
            %gradient matrices must be same size as the weights
            testCase.verifyEqual(size(testCase.model.gradients{3}), size(testCase.model.weights{3}));
            testCase.verifyEqual(size(testCase.model.gradients{2}), size(testCase.model.weights{2}));
            testCase.verifyEqual(size(testCase.model.gradients{1}), size(testCase.model.weights{1}));
        end

        function testForwardPassRelu(testCase)
            testCase.model.activation_fn = @utils.relu;
            testCase.model.final_activation_fn = @utils.softmax;
            testCase.model.forward_pass(testCase.inputs);
            testCase.verifyEqual(testCase.model.activations{2}, [0.74, 1.1192, 0.3564, 0.8744;0.5525, 0.8138, 0.1761, 0.6041], "AbsTol", 1e-4);
            testCase.verifyEqual(testCase.model.activations{3}, [1.96536 , 2.296904, 1.664372; 1.38873 , 1.858811, 1.166318], "AbsTol", 1e-4);
            testCase.verifyEqual(testCase.model.activations{4}, [0.47493816, 0.52506184;0.44214564, 0.55785436], "AbsTol", 1e-4);
        end

        function testBackwardPassReluWithRegularization(testCase)
            testCase.model.activation_fn = @utils.relu;
            testCase.model.final_activation_fn = @utils.softmax;
            testCase.model.regularization = 0.250;
            testCase.model.forward_pass(testCase.inputs);
            testCase.model.backprop(testCase.inputs, testCase.outputs, true);
            testCase.verifyEqual(testCase.model.deltas{3}, [-0.27506184, -0.45493816;-0.30785436,  0.27785436], "AbsTol", 1e-4);
            testCase.verifyEqual(testCase.model.deltas{2}, [-0.28479762, -0.54771722, -0.45969011;-0.24004786,  0.13466281,  0.0285567], "AbsTol", 1e-4);
            testCase.verifyEqual(testCase.model.deltas{1}, [-0.6782821 , -0.5171672 , -0.7658614 , -0.77661437;-0.08828193,  0.01617122, -0.16764972, -0.08642163], "AbsTol", 1e-4);
            testCase.verifyEqual(testCase.model.gradients{3}, [-0.37531106, -0.24162629;-0.54951686, -0.14548527;-0.34218066, -0.13030989], "AbsTol", 1e-4);
            testCase.verifyEqual(testCase.model.gradients{2}, [-0.08793834, -0.11295477, -0.09219655;-0.23954822, -0.22670826, -0.14562286; 0.04811285, -0.04574615,  0.00684764;-0.08826997, -0.08753707, -0.18110096], "AbsTol", 1e-4);
            testCase.verifyEqual(testCase.model.gradients{1}, [-0.12641214, -0.06353569, -0.16836246, -0.11637328; -0.18149873, -0.10817513, -0.20956937, -0.1799131 ], "AbsTol", 1e-4);
            testCase.verifyEqual(testCase.model.grad_biases{3}, [-0.2914581, -0.0885419], "AbsTol", 1e-4);
            testCase.verifyEqual(testCase.model.grad_biases{2}, [-0.26242274, -0.20652721, -0.2155667 ], "AbsTol", 1e-4);
            testCase.verifyEqual(testCase.model.grad_biases{1}, [-0.38328202, -0.25049799, -0.46675556, -0.431518 ], "AbsTol", 1e-4);
        end
    end
end