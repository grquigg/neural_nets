#include <iostream>
#include <cassert>
#include <string>
#include <vector>
// #include "utils.h"
#include "../include/utils.h"
#include "../include/lin_alg.h"
#include <chrono> 

float* transferMatrixToDevice(float **matrix, int height, int width) {
    float* deviceMatrix;
    cudaMalloc(&deviceMatrix, height*width*sizeof(float));
    for(int i = 0; i < height; i++) {
        cudaMemcpy(deviceMatrix+(i*width), matrix[i], sizeof(float)*width, cudaMemcpyHostToDevice);
    }
    return deviceMatrix;
}

int main(int argc, char** argv) {
    int numClasses = 10;
    // std::cout << "Hello World!" << std::endl;
    // std::cout << "Train data path: " << argv[1] << std::endl;
    if(argc != 5) {
        std::cout << "Need to specify paths for loading in the training data and the training labels" << std::endl;
        return 0;
    }
    std::string train_data_path = argv[1];
    std::string train_label_path = argv[2];
    std::string nWorkers_arg = argv[3];
    std::string nThreads_arg = argv[4];
    auto startInitial = std::chrono::system_clock::now();
    std::vector<std::vector<int>> inputs = readDataFromUByteFile(train_data_path);
    int size = inputs.size();
    int nFeatures = inputs[0].size();
    // std::cout << height << " " << width << std::endl;
    std::vector<float> input(size*nFeatures, 0.0f);
    for(int i = 0; i < size; i++) {
        for(int j = 0; j < nFeatures; j++) {
            input[i*nFeatures + j] = (float) inputs[i][j] / 255.0;
        }
    }
    // printMatrix(input, size, nFeatures);
    std::vector<std::vector<int>> outputs = readDataFromUByteFile(train_label_path);
    std::vector<float> weights = initializeRandomArray(nFeatures, numClasses);
    std::vector<float> product(size*nFeatures*10, 0.0);
    float learning_rate = 0.005;
    int nWorkers = std::stoi(nWorkers_arg);
    int nThreadsPerWorker = std::stoi(nThreads_arg);
    int BATCH_SIZE = size / (nWorkers * nThreadsPerWorker);
    int nEpochs = 10;
    // std::cout << "BATCH SIZE " << BATCH_SIZE << std::endl;
    float ** output;
    output = (float**)malloc(sizeof(float*) * size);
    for (int i = 0; i < size; i++) {
        output[i] = (float *)malloc(10*sizeof(float));
        for(int j = 0; j < 10; j++) {
            output[i][j] = 0;
        }
        output[i][outputs[i][0]] = 1.0;
    }
    float *d_outputs = transferMatrixToDevice(output, size, 10);

    //declare device variables
    float *d_inputs;
    float *d_weights;
    float *d_product;
    float *d_gradients;
    cudaMalloc(&d_gradients, nFeatures*numClasses*nWorkers*nThreadsPerWorker*sizeof(float));
    cudaMalloc(&d_inputs, size*nFeatures*sizeof(float));
    cudaMemcpy(d_inputs, input.data(), size*nFeatures*sizeof(float), cudaMemcpyHostToDevice);
    cudaMalloc(&d_weights, nFeatures*numClasses*sizeof(float));
    cudaMemcpy(d_weights, weights.data(), nFeatures*numClasses*sizeof(float), cudaMemcpyHostToDevice);
    cudaMalloc(&d_product, size*numClasses*sizeof(float));
    cudaMemcpy(d_product, product.data(), size*numClasses*sizeof(float), cudaMemcpyHostToDevice);

    //declare variables for computing metrics
    float *d_loss;
    int *d_correct;

    cudaMalloc(&d_loss, sizeof(float));
    cudaMalloc(&d_correct, sizeof(int));
    cudaMemcpy(d_loss, 0, sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_correct, 0, sizeof(int), cudaMemcpyHostToDevice);

    auto endInitial = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed_seconds = endInitial-startInitial;
    std::cout << "Finished initial setup in " << elapsed_seconds.count() << " seconds" << std::endl;
    auto startTrain = std::chrono::system_clock::now();
    for(int i = 0; i < nEpochs; i++) {
        //forward pass
        std::cout << "Starting forward pass..." << std::endl;
        auto initForward = std::chrono::system_clock::now();
        forward_pass<<<nWorkers, nThreadsPerWorker>>>(d_inputs, d_weights, d_outputs, d_product, d_gradients, BATCH_SIZE, nFeatures, numClasses, d_correct, d_loss);
        cudaDeviceSynchronize();
        auto endForward = std::chrono::system_clock::now();
        std::chrono::duration<double> elapsed_forward = endForward - initForward;
        std::cout << "Finished forward pass in " << elapsed_forward.count() << " seconds" << std::endl;
        int totalCorrect = 0;
        cudaMemcpy(&totalCorrect, d_correct, sizeof(int), cudaMemcpyDeviceToHost);
        float accuracy = (float) (totalCorrect) / size;
        printf("Accuracy: %f%%\n", accuracy*100);
        std::cout << "Starting backward pass..." << std::endl;
        auto initBackward = std::chrono::system_clock::now();
        ringReduce<<<nWorkers, nThreadsPerWorker>>>(d_gradients, nWorkers*nThreadsPerWorker, nFeatures*numClasses, (nFeatures*numClasses)/(nWorkers*nThreadsPerWorker));
        cudaDeviceSynchronize();
        backward_pass<<<1,1>>>(d_weights, d_gradients, 1, learning_rate, nFeatures, numClasses);
        cudaDeviceSynchronize();
        auto endBackward = std::chrono::system_clock::now();
        std::chrono::duration<double> elapsed_backward = endBackward - initBackward;
        std::cout << "Finished backward pass in " << elapsed_backward.count() << " seconds" << std::endl;
    }
    auto endTrain = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed = endTrain - startTrain;
    std::cout << "Finished training loop in " << elapsed.count() << " seconds" << std::endl;
    cudaFree(d_inputs);
    cudaFree(d_weights);
    cudaFree(d_product);
    cudaFree(d_gradients);

    return 0;
}