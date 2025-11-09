#include <gtest/gtest.h>
#include "../include/lin_alg.h"
#include "../include/utils.h"
#include "../include/models.h"

TEST(NeuralNetwork, TestActivationsForExOne) {
    int batch_size = 1;
    int * layers = new int[3]{1,2, 1};
    int nLayers = 2;
    float input[2] = {0.13000f, 0.42f};
    float **weights = new float*[2];
    weights[0] = new float[2]{0.1f, 0.2f};
    weights[1] = new float[2]{0.5f, 0.6f};
    float **biases = new float*[2];
    biases[0] = new float[2]{0.4f, 0.3f};
    biases[1] = new float[1]{0.7f};
    NeuralNetwork model(nLayers, layers, weights, biases, 1.0f);
    float correctOutput[3] = {0.601807f,0.58078581f, 0.79402745f};
    model.setupGPU(1, batch_size);
    model.final_activation = sigmoidHost;
    std::shared_ptr<float> d_input = transferMatrixToDevice(input, 1, 1);
    std::shared_ptr<float> activations = model.forward_pass(d_input, 1, 1, 1, 1);
    for(int i = 0; i < 3; i++) {
        EXPECT_FLOAT_EQ(activations.get()[i], correctOutput[i]);
    }
}

TEST(NeuralNetwork, TestActivationsForExTwo) {
    int batch_size = 1;
    int * layers = new int[3]{1,2, 1};
    int nLayers = 2;
    float input[2] = {0.13000f, 0.42f};
    float **weights = new float*[2];
    weights[0] = new float[2]{0.1f, 0.2f};
    weights[1] = new float[2]{0.5f, 0.6f};
    float **biases = new float*[2];
    biases[0] = new float[2]{0.4f, 0.3f};
    biases[1] = new float[1]{0.7f};
    NeuralNetwork model(nLayers, layers, weights, biases, 1.0f);
    float correctOutput[3] = {0.6087355f,0.59483749f, 0.79596603f};
    model.setupGPU(1, batch_size);
    model.final_activation = sigmoidHost;
    std::shared_ptr<float> d_input = transferMatrixToDevice(input+1, 1, 1);
    std::shared_ptr<float> activations = model.forward_pass(d_input, 1, 1, 1, 1);
    for(int i = 0; i < 3; i++) {
        EXPECT_FLOAT_EQ(activations.get()[i], correctOutput[i]);
    }
}

TEST(NeuralNetwork, TestFinalActivationsSoftmaxForExOne) {
    int batch_size = 1;
    int * layers = new int[3]{1,2, 1};
    int nLayers = 2;
    float input[2] = {0.13000f, 0.42f};
    float **weights = new float*[2];
    weights[0] = new float[2]{0.1f, 0.2f};
    weights[1] = new float[2]{0.5f, 0.6f};
    float **biases = new float*[2];
    biases[0] = new float[2]{0.4f, 0.3f};
    biases[1] = new float[1]{0.7f};
    NeuralNetwork model(nLayers, layers, weights, biases, 1.0f);
    float correctOutput[3] = {0.601807f,0.58078581f, 1.0f};
    model.setupGPU(1, batch_size);
    std::shared_ptr<float> d_input = transferMatrixToDevice(input, 1, 1);
    std::shared_ptr<float> activations = model.forward_pass(d_input, 1, 1, 1, 1);
    EXPECT_FLOAT_EQ(activations.get()[2], 1.0f);
}

TEST(NeuralNetwork, TestFinalActivationsSoftmaxForExTwo) {
    int batch_size = 1;
    int * layers = new int[3]{1,2, 1};
    int nLayers = 2;
    float input[2] = {0.13000f, 0.42f};
    float **weights = new float*[2];
    weights[0] = new float[2]{0.1f, 0.2f};
    weights[1] = new float[2]{0.5f, 0.6f};
    float **biases = new float*[2];
    biases[0] = new float[2]{0.4f, 0.3f};
    biases[1] = new float[1]{0.7f};
    NeuralNetwork model(nLayers, layers, weights, biases, 1.0f);
    float correctOutput[3] = {0.601807f,0.58078581f, 1.0f};
    model.setupGPU(1, batch_size);
    std::shared_ptr<float> d_input = transferMatrixToDevice(input+1, 1, 1);
    std::shared_ptr<float> activations = model.forward_pass(d_input, 1, 1, 1, 1);
    EXPECT_FLOAT_EQ(activations.get()[2], 1.0f);
}

TEST(NeuralNetwork, TestActivationsForBatchSizeTwo) {
    int batch_size = 2;
    int * layers = new int[3]{1,2, 1};
    int nLayers = 2;
    float input[2] = {0.13000f, 0.42f};
    float **weights = new float*[2];
    weights[0] = new float[2]{0.1f, 0.2f};
    weights[1] = new float[2]{0.5f, 0.6f};
    float **biases = new float*[2];
    biases[0] = new float[2]{0.4f, 0.3f};
    biases[1] = new float[1]{0.7f};
    NeuralNetwork model(nLayers, layers, weights, biases, 1.0f);
    float correctOutput[6] = {0.601807f,0.58078581f, 0.6087355f,0.59483749f, 0.79402745f, 0.79596603f};
    model.setupGPU(2, batch_size);
    model.final_activation = sigmoidHost;
    std::shared_ptr<float> d_input = transferMatrixToDevice(input, 2, 1);
    std::shared_ptr<float> activations = model.forward_pass(d_input, 2, 2, 2, 1);
    for(int i = 0; i < 6; i++) {
        EXPECT_FLOAT_EQ(activations.get()[i], correctOutput[i]);
    }
}

TEST(NeuralNetwork, TestDeltasForExOne) {
    int batch_size = 1;
    int * layers = new int[3]{1,2, 1};
    int nLayers = 2;
    float input[2] = {0.13000f, 0.42f};
    float output[2] = {0.9f, 0.23f};
    float **weights = new float*[2];
    weights[0] = new float[2]{0.1f, 0.2f};
    weights[1] = new float[2]{0.5f, 0.6f};
    float **biases = new float*[2];
    biases[0] = new float[2]{0.4f, 0.3f};
    biases[1] = new float[1]{0.7f};
    NeuralNetwork model(nLayers, layers, weights, biases, 1.0f);
    model.setupGPU(1, batch_size);
    model.final_activation = sigmoidHost;
    std::shared_ptr<float> d_input = transferMatrixToDevice(input, batch_size, 1);
    std::shared_ptr<float> d_y = transferMatrixToDevice(output, 1, 2);

    model.forward_pass(d_input, batch_size, batch_size, 1, 1);
    model.backprop(batch_size, d_input, d_y);
    float **correctGradients = new float*[model.nLayers];
    float **correctDeltas = new float*[model.nLayers];
    correctGradients[0] = new float[2]{-0.0016506595f, -0.0020125185f};
    correctGradients[1] = new float[2]{-0.06377501f, -0.061547343f};
    correctDeltas[0] = new float[2]{-0.012697381f, -0.015480912f};
    correctDeltas[1] = new float[1]{-0.10597253f};
    float **deltas = new float*[model.nLayers];
    float **gradients = new float*[model.nLayers];
    for(int i = 0; i < model.nLayers; i++) {
        std::cout << "Batch: " << batch_size*model.layer_size[i+1] << std::endl;
        deltas[i] = new float[batch_size*model.layer_size[i+1]];
        gradients[i] = new float[model.layer_size[i]*model.layer_size[i+1]];
        cudaMemcpy(deltas[i], model.deltas[i], batch_size*model.layer_size[i+1]*sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(gradients[i], model.gradients[i], model.layer_size[i]*model.layer_size[i+1]*sizeof(float), cudaMemcpyDeviceToHost);
        for(int j = 0; j < batch_size*model.layer_size[i+1]; j++) {
            EXPECT_FLOAT_EQ(deltas[i][j], correctDeltas[i][j]);
        }
        for(int j = 0; j < model.layer_size[i]*model.layer_size[i+1]; j++) {
            EXPECT_FLOAT_EQ(gradients[i][j], correctGradients[i][j]);
        }    
    }
}

TEST(NeuralNetwork, TestDeltasForExTwo) {
    int batch_size = 1;
    int * layers = new int[3]{1,2, 1};
    int nLayers = 2;
    float input[2] = {0.13000f, 0.42f};
    float output[2] = {0.9f, 0.23f};
    float **weights = new float*[2];
    weights[0] = new float[2]{0.1f, 0.2f};
    weights[1] = new float[2]{0.5f, 0.6f};
    float **biases = new float*[2];
    biases[0] = new float[2]{0.4f, 0.3f};
    biases[1] = new float[1]{0.7f};
    NeuralNetwork model(nLayers, layers, weights, biases, 1.0f);
    model.setupGPU(1, batch_size);
    model.final_activation = sigmoidHost;
    std::shared_ptr<float> d_input = transferMatrixToDevice(input+1, batch_size, 1);
    std::shared_ptr<float> d_y = transferMatrixToDevice(output+1, 1, 2);

    model.forward_pass(d_input, batch_size, batch_size, 1, 1);
    model.backprop(batch_size, d_input, d_y);
    float **correctGradients = new float*[model.nLayers];
    float **correctDeltas = new float*[model.nLayers];
    correctGradients[0] = new float[2]{0.028307969f, 0.034373082f};
    correctGradients[1] = new float[2]{0.34452361f, 0.33665779f};
    correctDeltas[0] = new float[2]{0.067399926f, 0.081840672f};
    correctDeltas[1] = new float[1]{0.56596601f};
    float **deltas = new float*[model.nLayers];
    float **gradients = new float*[model.nLayers];
    for(int i = 0; i < model.nLayers; i++) {
        std::cout << "Batch: " << batch_size*model.layer_size[i+1] << std::endl;
        deltas[i] = new float[batch_size*model.layer_size[i+1]];
        gradients[i] = new float[model.layer_size[i]*model.layer_size[i+1]];
        cudaMemcpy(deltas[i], model.deltas[i], batch_size*model.layer_size[i+1]*sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(gradients[i], model.gradients[i], model.layer_size[i]*model.layer_size[i+1]*sizeof(float), cudaMemcpyDeviceToHost);
        for(int j = 0; j < batch_size*model.layer_size[i+1]; j++) {
            EXPECT_FLOAT_EQ(deltas[i][j], correctDeltas[i][j]);
        }
        for(int j = 0; j < model.layer_size[i]*model.layer_size[i+1]; j++) {
            EXPECT_FLOAT_EQ(gradients[i][j], correctGradients[i][j]);
        }    
    }
}

TEST(NeuralNetwork, TestGradients) {
    int batch_size = 2;
    int * layers = new int[3]{1,2, 1};
    int nLayers = 2;
    int nWorkers = 2;
    int nThreadsPerWorker = 1;
    float input[2] = {0.13000f, 0.42f};
    float output[2] = {0.9f, 0.23f};
    float **weights = new float*[2];
    weights[0] = new float[2]{0.1f, 0.2f};
    weights[1] = new float[2]{0.5f, 0.6f};
    float **biases = new float*[2];
    biases[0] = new float[2]{0.4f, 0.3f};
    biases[1] = new float[1]{0.7f};
    NeuralNetwork model(nLayers, layers, weights, biases, 1.0f);
    model.setupGPU(nWorkers*nThreadsPerWorker, batch_size);
    model.final_activation = sigmoidHost;
    std::shared_ptr<float> d_input = transferMatrixToDevice(input, batch_size, 1);
    std::shared_ptr<float> d_y = transferMatrixToDevice(output, 2, 1);

    model.forward_pass(d_input, batch_size, batch_size, nWorkers, nThreadsPerWorker);
    model.backprop(batch_size, d_input, d_y);
    float **correctGradients = new float*[model.nLayers];
    correctGradients[0] = new float[2]{0.013328655f, 0.016180281f};
    correctGradients[1] = new float[2]{0.1403743f, 0.13755523f};
    float **gradients = new float*[model.nLayers];
    for(int i = 0; i < model.nLayers; i++) {
        std::cout << "Batch: " << batch_size*model.layer_size[i+1] << std::endl;
        gradients[i] = new float[model.layer_size[i]*model.layer_size[i+1]];
        cudaMemcpy(gradients[i], model.gradients[i], model.layer_size[i]*model.layer_size[i+1]*sizeof(float), cudaMemcpyDeviceToHost);
        for(int j = 0; j < model.layer_size[i]*model.layer_size[i+1]; j++) {
            EXPECT_FLOAT_EQ(gradients[i][j], correctGradients[i][j]);
        }    
    }
}
