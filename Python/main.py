from neural.utils import computeAccuracy, read_file, relu, cross_entropy_loss, softMax
from neural.nn import NeuralNetwork
import numpy as np
np.random.seed(1)

if __name__ == "__main__":
    input_path = "../mnist/train-images.idx3-ubyte"
    output_path = "../mnist/train-labels.idx1-ubyte"
    BATCH_SIZE = 4000
    FULL_SIZE = 60000
    regularizer = 1
    learning_rate = 0.001
    epochs = 1000
    layers = [784, 64, 10]
    # Initialize inputs array with zeros and then set specific indices to 1.0
    inputs = read_file(input_path)
    inputs = np.reshape(inputs, (60000, 784)).astype(float)
    inputs /= 255.0
    # Initialize outputs array with zeros and then set specific indices to 1.0
    outputs = read_file(output_path)
    output_one_hot = np.zeros((60000, 10))
    for i in range(len(outputs)):
        output_one_hot[i][outputs[i]] = 1
    weights = []
    biases = []
    for i in range(len(layers)-1):
        weights.append(np.random.randn(layers[i], layers[i+1])*0.01)
        biases.append(np.random.randn(1, layers[i+1])*0.01)

    model = NeuralNetwork(layers, biases, weights, learning_rate=learning_rate, activation_fn=relu, final_activation=softMax, regularizer=regularizer)
    for epoch in range(epochs):
        accuracy = 0
        numCorrect = 0
        logLoss = 0
        for i in range(0, FULL_SIZE, BATCH_SIZE):
            predicted = model.forward_prop(inputs[i:i+BATCH_SIZE])
            numCorrect += computeAccuracy(predicted, output_one_hot[i:i+BATCH_SIZE])
            logLoss += cross_entropy_loss(output_one_hot[i:i+BATCH_SIZE], predicted)
            model.backprop(predicted, output_one_hot[i:i+BATCH_SIZE], regularize=False)
            model.update_weights()
        accuracy = numCorrect / FULL_SIZE * 100
        print(f"Epoch {epoch}\tAccuracy {accuracy}%\t\tLog loss {logLoss / (FULL_SIZE / BATCH_SIZE)}")