classdef ExampleTestCases1 < matlab.unittest.TestCase
    properties
        nLayers = 3;
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
            testCase.layers = [1,2,1];
            testCase.weights = cell(2,1);
            testCase.weights{1} = [0.1; 0.2];
            testCase.weights{2} = [0.5, 0.6];
            testCase.biases = cell(2,1);
            testCase.biases{1} = [0.4, 0.3];
            testCase.biases{2} = [0.7];
            testCase.model = NeuralNetwork(testCase.layers, testCase.weights, testCase.biases);
            testCase.inputs = [0.13;0.42];
            testCase.outputs = [0.9;0.23];
            testCase.activations = cell(2,1);
        end

    end

    methods(TestMethodTeardown)
        function closeFigure(testCase)
        end
    end
    
    methods(Test)
        function dotProduct1(testCase)
            % Test to check some default properties of the figure
            testCase.activations{1} = (testCase.inputs * testCase.model.weights{1}') + testCase.model.biases{1};
            expected = [0.4130,0.3260;0.4420,0.3840];
            testCase.verifyEqual(testCase.activations{1}, expected, "AbsTol", 1e-5);
        end

        function sigmoid1(testCase)
            testCase.activations{1} = (testCase.inputs * testCase.model.weights{1}') + testCase.model.biases{1};
            testCase.activations{1} = utils.sigmoid(testCase.activations{1});
            expected = [0.60181,0.58079;0.60874,0.59484];
            testCase.verifyEqual(testCase.activations{1}, expected, "AbsTol", 1e-5);
        end

        function dotProduct2(testCase)
            testCase.activations{1} = (testCase.inputs * testCase.model.weights{1}') + testCase.model.biases{1};
            testCase.activations{1} = utils.sigmoid(testCase.activations{1});
            testCase.activations{2} = (testCase.activations{1} * testCase.model.weights{2}') + testCase.model.biases{2}; 
            expected = [1.34937; 1.36127];
            testCase.verifyEqual(testCase.activations{2}, expected, "AbsTol", 1e-5);
        end

        function sigmoidProduct2(testCase)
            testCase.activations{1} = (testCase.inputs * testCase.model.weights{1}') + testCase.model.biases{1};
            testCase.activations{1} = utils.sigmoid(testCase.activations{1});
            testCase.activations{2} = (testCase.activations{1} * testCase.model.weights{2}') + testCase.model.biases{2}; 
            testCase.activations{2} = utils.sigmoid(testCase.activations{2});
            expected = [0.79403; 0.79597];
            testCase.verifyEqual(testCase.activations{2}, expected, "AbsTol", 1e-5);
        end

        function softmaxProduct2(testCase)
            testCase.activations{1} = (testCase.inputs * testCase.model.weights{1}') + testCase.model.biases{1};
            testCase.activations{1} = utils.sigmoid(testCase.activations{1});
            testCase.activations{2} = (testCase.activations{1} * testCase.model.weights{2}') + testCase.model.biases{2}; 
            testCase.activations{2} = utils.softmax(testCase.activations{2});
            expected = [1.0; 1.0];
            testCase.verifyEqual(testCase.activations{2}, expected, "AbsTol", 1e-5);
        end

        function forwardPassTestSigmoidFinalLayer(testCase)
            testCase.model.forward_pass(testCase.inputs);
            expected = [0.79403; 0.79597];
            testCase.verifyEqual(testCase.model.activations{2}, expected, "AbsTol", 1e-5);
        end

        function forwardPassTestSoftmaxFinalLayer(testCase)
            testCase.model.activation_fn = @utils.softmax;
            testCase.model.forward_pass(testCase.inputs);
            expected = [1.0; 1.0];
            testCase.verifyEqual(testCase.model.activations{2}, expected, "AbsTol", 1e-5);
        end

        function logLossSoftmax(testCase)
            testCase.model.activation_fn = @utils.softmax;
            testCase.model.forward_pass(testCase.inputs);
            loss = utils.crossEntropyLoss(testCase.outputs, testCase.model.activations{2});
            testCase.verifyEqual(loss, -2.56394985712845, "AbsTol", 1e-5);
        end
    end
end