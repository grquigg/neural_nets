#include <iostream>
#include <cassert>
#include <string>
#include <vector>
// #include "utils.h"
#include "../include/utils.h"
#include "../include/lin_alg.h"


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
    std::cout << "Hello World!" << std::endl;
    std::cout << "Train data path: " << argv[1] << std::endl;
    if(argc != 3) {
        std::cout << "Need to specify paths for loading in the training data and the training labels" << std::endl;
        return 0;
    }
    std::string train_data_path = argv[1];
    std::string train_label_path = argv[2];
    std::cout << "args" << std::endl;
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
    std::vector<std::vector<int>> outputs = readDataFromUByteFile(train_label_path);
    std::vector<float> weights = initializeRandomArray(nFeatures, numClasses);
    std::vector<float> product(size*nFeatures*10, 0.0);
    float learning_rate = 0.005;
    int nWorkers = 4;
    int nThreadsPerWorker = 2;
    int BATCH_SIZE = size / (nWorkers * nThreadsPerWorker);

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
    cudaMalloc(&d_inputs, size*nFeatures*sizeof(float));
    cudaMemcpy(d_inputs, input.data(), size*nFeatures*sizeof(float), cudaMemcpyHostToDevice);
    cudaMalloc(&d_weights, nFeatures*numClasses*sizeof(float));
    cudaMemcpy(d_weights, weights.data(), nFeatures*numClasses*sizeof(float), cudaMemcpyHostToDevice);
    cudaMalloc(&d_product, size*numClasses*sizeof(float));
    cudaMemcpy(d_product, product.data(), size*numClasses*sizeof(float), cudaMemcpyHostToDevice);
    forward_pass<<<nWorkers, nThreadsPerWorker>>>(d_inputs, d_weights, d_outputs, d_product, BATCH_SIZE, nFeatures, numClasses);
    cudaDeviceSynchronize();

    float *produce = (float*)malloc(size*numClasses*sizeof(float));
    cudaMemcpy(produce, d_product, size*numClasses*sizeof(float), cudaMemcpyDeviceToHost);
    printMatrix(produce, size, numClasses);
    cudaFree(d_inputs);
    cudaFree(d_weights);
    cudaFree(d_product);

    return 0;
}