classdef NeuralNetwork < handle
    properties
        nLayers = 0;
        layer_size = [];
        weights;
        biases;
        activations;
        activation_fn;
        deltas;
    end
    methods
        function obj = NeuralNetwork(layer_size, weights, biases)
            obj.layer_size = layer_size;
            obj.nLayers = length(layer_size)-1;
            obj.weights = weights;
            obj.biases = biases;
            obj.activations = cell(obj.nLayers,1);
            obj.activation_fn = @utils.sigmoid;
        end

        function obj = forward_pass(obj, inputs)
            obj.activations{1} = (inputs * obj.weights{1}') + obj.biases{1};
            obj.activations{1} = utils.sigmoid(obj.activations{1});
            for i=1:obj.nLayers-1
                obj.activations{i+1} = (obj.activations{i} * obj.weights{i+1}') + obj.biases{i+1};
                obj.activations{i+1} = utils.sigmoid(obj.activations{i+1});
            end
            obj.activations{obj.nLayers} = (obj.activations{obj.nLayers-1} * obj.weights{obj.nLayers}') + obj.biases{obj.nLayers};
            obj.activations{obj.nLayers} = obj.activation_fn(obj.activations{obj.nLayers});
        end
    end
end