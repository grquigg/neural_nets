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
            testCase.weights{1} = [0.1,0.2];
            testCase.weights{2} = [0.5;0.6];
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
            testCase.activations{1} = (testCase.inputs * testCase.model.weights{1}) + testCase.model.biases{1};
            expected = [0.4130,0.3260;0.4420,0.3840];
            testCase.verifyEqual(testCase.activations{1}, expected, "AbsTol", 1e-5);
        end

        function sigmoid1(testCase)
            testCase.activations{1} = (testCase.inputs * testCase.model.weights{1}) + testCase.model.biases{1};
            testCase.activations{1} = utils.sigmoid(testCase.activations{1});
            expected = [0.60181,0.58079;0.60874,0.59484];
            testCase.verifyEqual(testCase.activations{1}, expected, "AbsTol", 1e-5);
        end

        function dotProduct2(testCase)
            testCase.activations{1} = (testCase.inputs * testCase.model.weights{1}) + testCase.model.biases{1};
            testCase.activations{1} = utils.sigmoid(testCase.activations{1});
            testCase.activations{2} = (testCase.activations{1} * testCase.model.weights{2}) + testCase.model.biases{2}; 
            expected = [1.34937; 1.36127];
            testCase.verifyEqual(testCase.activations{2}, expected, "AbsTol", 1e-5);
        end

        function sigmoidProduct2(testCase)
            testCase.activations{1} = (testCase.inputs * testCase.model.weights{1}) + testCase.model.biases{1};
            testCase.activations{1} = utils.sigmoid(testCase.activations{1});
            testCase.activations{2} = (testCase.activations{1} * testCase.model.weights{2}) + testCase.model.biases{2}; 
            testCase.activations{2} = utils.sigmoid(testCase.activations{2});
            expected = [0.79403; 0.79597];
            testCase.verifyEqual(testCase.activations{2}, expected, "AbsTol", 1e-5);
        end

        function softmaxProduct2(testCase)
            testCase.activations{1} = (testCase.inputs * testCase.model.weights{1}) + testCase.model.biases{1};
            testCase.activations{1} = utils.sigmoid(testCase.activations{1});
            testCase.activations{2} = (testCase.activations{1} * testCase.model.weights{2}) + testCase.model.biases{2}; 
            testCase.activations{2} = utils.softmax(testCase.activations{2});
            expected = [1.0; 1.0];
            testCase.verifyEqual(testCase.activations{2}, expected, "AbsTol", 1e-5);
        end

        function forwardPassTestSigmoidFinalLayer(testCase)
            testCase.model.forward_pass(testCase.inputs);
            expected = [0.79403; 0.79597];
            testCase.verifyEqual(testCase.model.activations{3}, expected, "AbsTol", 1e-5);
        end

        function forwardPassTestSoftmaxFinalLayer(testCase)
            testCase.model.final_activation_fn = @utils.softmax;
            testCase.model.forward_pass(testCase.inputs);
            expected = [1.0; 1.0];
            testCase.verifyEqual(testCase.model.activations{3}, expected, "AbsTol", 1e-5);
        end

        %test that everything works as expected with a final sigmoid
        %activation function rather than softmax
        function testBackpropDeltaSigmoid(testCase)
            testCase.model.activation_fn = @utils.sigmoid;
            testCase.model.forward_pass(testCase.inputs);
            testCase.model.backprop(testCase.inputs, testCase.outputs, false);
            %verify that the calculated deltas are correct
            testCase.verifyEqual(testCase.model.deltas{2}, [-0.10597257;0.56596607], "AbsTol", 1e-6);
            testCase.verifyEqual(testCase.model.deltas{1}, [-0.01270,-0.01548;0.06740,0.08184], "AbsTol", 1e-5);
            %gradient matrices must be same size as the weights
            testCase.verifyEqual(size(testCase.model.gradients{2}), [2,1]);
            testCase.verifyEqual(size(testCase.model.gradients{1}), [1,2]);
            %verify that the calculated gradients are correct
            testCase.verifyEqual(testCase.model.gradients{2}, [0.14037;0.13756], "AbsTol", 1e-5);
            testCase.verifyEqual(testCase.model.gradients{1}, [0.01333,0.01618], "AbsTol", 1e-5);
        end

        function testForwardPassRelu(testCase)
            testCase.model.activation_fn = @utils.relu;
            testCase.model.forward_pass(testCase.inputs);
            testCase.verifyEqual(testCase.model.activations{2}, [0.4130,0.3260;0.4420,0.3840], "AbsTol", 1e-3);
            testCase.verifyEqual(testCase.model.activations{3}, [0.75065;0.75976654], "AbsTol", 1e-3);
        end


        function testBackwardPassRelu(testCase)
            testCase.model.activation_fn = @utils.relu;
            testCase.model.forward_pass(testCase.inputs);
            testCase.model.backprop(testCase.inputs, testCase.outputs, false);
            testCase.verifyEqual(testCase.model.deltas{2}, [-0.14934662;0.52976654], "AbsTol", 1e-5);
            testCase.verifyEqual(testCase.model.deltas{1}, [-0.07467331,-0.08960797;0.26488327,0.31785992], "AbsTol", 1e-5);
            testCase.verifyEqual(testCase.model.gradients{2}, [0.08623833;0.07737168], "AbsTol", 1e-5);
            testCase.verifyEqual(testCase.model.gradients{1}, [0.05077172,0.06092607], "AbsTol", 1e-5);
            testCase.verifyEqual(size(testCase.model.gradients{1}), size(testCase.model.weights{1}));
            testCase.verifyEqual(size(testCase.model.gradients{2}), size(testCase.model.weights{2}));
        end
    end
end