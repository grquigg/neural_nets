classdef NeuralNetwork < handle
    properties
        nLayers = 0;
        layer_size = [];
        weights;
        biases;
        activations;
        activation_fn;
        final_activation_fn;
        deltas;
        gradients;
        grad_biases;
    end
    methods
        function obj = NeuralNetwork(layer_size, weights, biases)
            obj.layer_size = layer_size;
            obj.nLayers = length(layer_size)-1;
            obj.weights = weights;
            obj.biases = biases;
            obj.activations = cell(obj.nLayers+1,1);
            obj.deltas = cell(obj.nLayers,1);
            obj.gradients = cell(obj.nLayers, 1);
            obj.grad_biases = cell(obj.nLayers, 1);
            obj.activation_fn = @utils.sigmoid;
            obj.final_activation_fn = @utils.sigmoid;
        end

        function forward_pass(obj, inputs)
            obj.activations{1} = inputs;
            for i=2:obj.nLayers
                obj.activations{i} = (obj.activations{i-1} * obj.weights{i-1}) + obj.biases{i-1};
                obj.activations{i} = obj.activation_fn(obj.activations{i});
            end
            obj.activations{obj.nLayers+1} = (obj.activations{obj.nLayers} * obj.weights{obj.nLayers}) + obj.biases{obj.nLayers};
            obj.activations{obj.nLayers+1} = obj.final_activation_fn(obj.activations{obj.nLayers+1});
        end

        function backprop(obj, inputs, outputs)
            L = obj.nLayers+1;
            obj.deltas{obj.nLayers} = obj.activations{obj.nLayers+1} - outputs;
            obj.gradients{L-1} = obj.activations{obj.nLayers}' * obj.deltas{obj.nLayers} ./ size(inputs, 1);
            obj.grad_biases{L-1} = sum(obj.deltas{obj.nLayers})./ size(inputs, 1);
            for i=L-2:-1:1
                obj.deltas{i} = obj.deltas{i+1} * obj.weights{i+1}';
                derivative = utils.getDerivative(obj.activation_fn);
                obj.deltas{i} = obj.deltas{i} .* derivative(obj.activations{i+1});
                obj.gradients{i} = (obj.activations{i}' * obj.deltas{i}) ./ size(inputs, 1);
                obj.grad_biases{i} = sum(obj.deltas{i}) ./ size(inputs, 1);
            end
        end

        function updateWeights(obj, learning_rate)
            for i=1:obj.nLayers-1
                obj.weights{i} = obj.weights{i} - (obj.gradients{i} * learning_rate);
                obj.biases{i} = obj.biases{i} - (obj.grad_biases{i} * learning_rate);
            end
        end
    end
end