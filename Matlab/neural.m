% Initialize file names
input_data_path = "../mnist/train-images.idx3-ubyte";
input_labels_path = "../mnist/train-labels.idx1-ubyte";
test_data_path = "../mnist/t10k-images.idx3-ubyte";
test_labels_path = "../mnist/t10k-labels.idx1-ubyte";

% results = runtests('ExampleTestCases1.m');

% BATCH_SIZE = 5000;
% EPOCHS = 10;
% learning_rate = 0.005;
% data = readInUbyteFile(input_data_path);
% data = data ./ 255.0;
% outputs = readInUbyteFile(input_labels_path);
% output = zeros(size(outputs,1),10);
% 
% test_data = readInUbyteFile(test_data_path);
% test_data = test_data ./ 255.0;
% test_labels = readInUbyteFile(test_labels_path);
% label = zeros(size(test_labels, 1), 10);
% 
% for i=1:size(outputs,1)
%     output(i, outputs(i,1)+1) = 1.0;
% end
% 
% for i=1:size(label, 1)
%     label(i, test_labels(i,1)+1) = 1.0;
% end
% 
% weights = ones(size(data,2), 10);
% weights = weights * 0.25;
% for i=1:EPOCHS
%     nCorrect = 0;
%     accuracy = 0;
%     for j= 1:BATCH_SIZE:size(data,1)
%         product = data(j:j+BATCH_SIZE-1, :) * weights;
%         probs = softmax(product);
%         nCorrect = nCorrect + getPredictions(probs, outputs(j:j+BATCH_SIZE-1, :));
%         grads = output(j:j+BATCH_SIZE-1, :) - probs;
%         delta_w = data(j:j+BATCH_SIZE-1, :)' * grads;
%         weights = weights - (learning_rate .* delta_w ./ BATCH_SIZE);
%     end
%     accuracy = nCorrect / size(data,1);
%     fprintf("Accuracy: %f%%\n", accuracy*100);
% end
% 
% 
% function correct = getPredictions(predicted, actual)
%     [~, predictions] = max(predicted, [], 2);
%     valid = predictions == actual;
%     correct = sum(valid);
% end
% 
% function data = readInUbyteFile(fileName)
%     fileID = fopen(fileName, 'r');
%     magic = fread(fileID, 4, 'ubit8');
%     dims = magic(4);
%     dimensions = fread(fileID, dims, 'uint32', 'ieee-be');
%     val = prod(dimensions(2:size(dimensions)))
%     data = fread(fileID, [dimensions(1), val], 'ubit8');
%     fclose(fileID);
% end
% 
% function probs = softmax(arr)
%     logits = exp(arr);
%     total = log(sum(logits, 2));
%     probs = exp(arr - total);
% end