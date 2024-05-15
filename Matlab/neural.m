% Initialize file names
input_data_path = "../mnist/train-images.idx3-ubyte";
input_labels_path = "../mnist/train-labels.idx1-ubyte";
test_data_path = "../mnist/t10k-images.idx3-ubyte";
test_labels_path = "../mnist/t10k-labels.idx1-ubyte";

BATCH_SIZE = 4000;
EPOCHS = 10;
layers = [784, 64, 64, 10];
learning_rate = 0.001;
data = utils.readInUbyteFile(input_data_path);
data = data ./ 255.0;
size(data)
outputs = utils.readInUbyteFile(input_labels_path);
output = zeros(size(outputs,1),10);

test_data = utils.readInUbyteFile(test_data_path);
test_data = test_data ./ 255.0;
test_labels = utils.readInUbyteFile(test_labels_path);
label = zeros(size(test_labels, 1), 10);

for i=1:size(outputs,1)
    output(i, outputs(i,1)+1) = 1.0;
end

for i=1:size(label, 1)
    label(i, test_labels(i,1)+1) = 1.0;
end

%create weight and bias matrices
weights = cell(layers-1);
biases = cell(layers-1);
for i=1:size(layers, 2)-1
    weights{i} = randn(layers(i), layers(i+1));
    biases{i} = randn(1, layers(i+1));
end
model = NeuralNetwork(layers, weights, biases);
model.activation_fn = @utils.relu;
model.final_activation_fn = @utils.softmax;
for i=1:EPOCHS
    nCorrect = 0;
    accuracy = 0;
    loss = 0;
    for j= 1:BATCH_SIZE:size(data,1)
        model.forward_pass(data(j:j+BATCH_SIZE-1, :));
        probs = model.activations{end};
        nCorrect = nCorrect + getPredictions(probs, outputs(j:j+BATCH_SIZE-1, :));
        loss = loss + utils.crossEntropyLoss(outputs(j:j+BATCH_SIZE-1, :), probs);
        model.backprop(probs, outputs(j:j+BATCH_SIZE-1, :));
        model.updateWeights(learning_rate);
    end
    accuracy = nCorrect / size(data,1);
    fprintf("Accuracy: %f%%\tLog Loss: %f%\n\n", accuracy*100, loss);
end


function correct = getPredictions(predicted, actual)
    [~, predictions] = max(predicted, [], 2);
    valid = predictions == actual;
    correct = sum(valid);
end