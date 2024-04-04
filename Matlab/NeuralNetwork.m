classdef NeuralNetwork < handle
    properties
        nLayers = 0;
        layer_size = [];
        weights;
        biases;
        activations;
        activation_fn;
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
            obj.activations = cell(obj.nLayers,1);
            obj.deltas = cell(obj.nLayers,1);
            obj.gradients = cell(obj.nLayers, 1);
            obj.grad_biases = cell(obj.nLayers, 1);
            obj.activation_fn = @utils.sigmoid;
        end

        function forward_pass(obj, inputs)
            obj.activations{1} = (inputs * obj.weights{1}') + obj.biases{1};
            obj.activations{1} = utils.sigmoid(obj.activations{1});
            for i=1:obj.nLayers-1
                obj.activations{i+1} = (obj.activations{i} * obj.weights{i+1}') + obj.biases{i+1};
                obj.activations{i+1} = utils.sigmoid(obj.activations{i+1});
            end
            obj.activations{obj.nLayers} = (obj.activations{obj.nLayers-1} * obj.weights{obj.nLayers}') + obj.biases{obj.nLayers};
            obj.activations{obj.nLayers} = obj.activation_fn(obj.activations{obj.nLayers});
        end

        function backprop(obj, inputs, outputs)
            obj.deltas{obj.nLayers} = obj.activations{obj.nLayers} - outputs;
            obj.grad_biases{obj.nLayers} = obj.deltas{obj.nLayers};
            for i=obj.nLayers-1:-1:1
                obj.gradients{i+1} = (obj.deltas{i+1}' * obj.activations{i}) ./ size(inputs, 1);
                obj.deltas{i} = obj.deltas{i+1} * obj.weights{i+1};
                obj.deltas{i} = obj.deltas{i} .* ((1 - obj.activations{i}) .* obj.activations{i});
            end
            obj.gradients{1} = (obj.deltas{1}' * inputs) ./ size(inputs, 1);
        end
    end
end