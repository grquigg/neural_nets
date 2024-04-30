from neural.utils import computeAccuracy, read_file, relu, softMax
from neural.nn import NeuralNetwork
import numpy as np
np.random.seed(1)

if __name__ == "__main__":
    input_train_path = "../mnist/train-images.idx3-ubyte"
    output_train_path = "../mnist/train-labels.idx1-ubyte"
    input_test_path = "../mnist/t10k-images.idx3-ubyte"
    output_test_path = "../mnist/t10k-labels.idx1-ubyte"
    BATCH_SIZE = 4000
    TEST_BATCH_SIZE = 5000
    FULL_SIZE = 60000
    regularizer = 1
    learning_rate = 0.01
    epochs = 400
    layers = [784, 64, 10]
    # Initialize inputs array with zeros and then set specific indices to 1.0
    inputs = read_file(input_train_path)
    inputs = np.reshape(inputs, (60000, 784)).astype(float)
    inputs /= 255.0
    # Initialize outputs array with zeros and then set specific indices to 1.0
    outputs = read_file(output_train_path)
    output_one_hot = np.zeros((60000, 10))
    for i in range(len(outputs)):
        output_one_hot[i][outputs[i]] = 1
    ###TESTING DATA###
    test_inputs = read_file(input_test_path)
    test_inputs = np.reshape(test_inputs, (10000,784)).astype(float)
    test_inputs /= 255.0

    test_outputs = read_file(output_test_path)
    test_output_one_hot = np.zeros((10000,10))
    for i in range(len(test_outputs)):
        test_output_one_hot[i][test_outputs[i]] += 1

    weights = []
    biases = []
    for i in range(len(layers)-1):
        weights.append(np.random.randn(layers[i], layers[i+1])*0.01)
        biases.append(np.random.randn(1, layers[i+1])*0.01)

    model = NeuralNetwork(layers, biases, weights, learning_rate=learning_rate, activation_fn=relu, final_activation=softMax, regularizer=regularizer)
    for epoch in range(epochs):
        testAccuracy = 0
        numTestCorrect = 0
        accuracy = 0
        numCorrect = 0
        logLoss = 0
        for i in range(0, FULL_SIZE, BATCH_SIZE):
            predicted = model.forward_prop(inputs[i:i+BATCH_SIZE])
            numCorrect += computeAccuracy(predicted, output_one_hot[i:i+BATCH_SIZE])
            logLoss += model.cost(predicted, output_one_hot[i:i+BATCH_SIZE], regularize=True)
            model.backprop(predicted, output_one_hot[i:i+BATCH_SIZE], regularize=True)
            model.update_weights()
        for i in range(0, 10000, TEST_BATCH_SIZE):
            predictions = model.forward_prop(test_inputs[i:i+TEST_BATCH_SIZE])
            numTestCorrect += computeAccuracy(predictions, test_output_one_hot[i:i+TEST_BATCH_SIZE])
        accuracy = numCorrect / FULL_SIZE * 100
        testAccuracy = numTestCorrect / 10000 * 100
        print(f"Epoch {epoch+1:3d}\tTrain Accuracy: {accuracy:7.4f}%\tTest Accuracy: {testAccuracy:7.4f}%\t\tLog loss {logLoss / (FULL_SIZE / BATCH_SIZE):.5f}")