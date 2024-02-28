#include <iostream>
#include <cassert>
#include <string>
#include <vector>
// #include "utils.h"
#include "../include/utils.h"
#include "../include/lin_alg.h"
#include <chrono> 

float* transferMatrixToDevice(float *matrix, int height, int width) {
    float* deviceMatrix;
    cudaMalloc(&deviceMatrix, height*width*sizeof(float));
    for(int i = 0; i < height; i++) {
        cudaMemcpy(deviceMatrix+(i*width), matrix+(i*width), sizeof(float)*width, cudaMemcpyHostToDevice);
    }
    return deviceMatrix;
}

int main(int argc, char** argv) {
    // std::cout << "Hello World!" << std::endl;
    // std::cout << "Train data path: " << argv[1] << std::endl;
    // if(argc != 5) {
    //     std::cout << "Need to specify paths for loading in the training data and the training labels" << std::endl;
    //     return 0;
    // }
    std::string train_data_path = argv[1];
    std::string train_label_path = argv[2];
    std::string test_data_path = argv[3];
    std::string test_label_path = argv[4];
    // std::string nWorkers_arg = argv[5];
    // std::string nThreads_arg = argv[6];
    int numClasses = 10;
    int BATCH_SIZE = 4000;
    int nEpochs = 100;
    float learning_rate = 0.01;
    int nWorkers = 16;
    int nThreadsPerWorker = 10;

    //read in input file
    //auto startInitial = std::chrono::system_clock::now();
    std::vector<std::vector<int>> inputs = readDataFromUByteFile(train_data_path);
    std::vector<std::vector<int>> test_inputs = readDataFromUByteFile(test_data_path);
    int size = inputs.size();
    int nFeatures = inputs[0].size();
    // std::cout << height << " " << width << std::endl;
    std::vector<float> input(size*nFeatures, 0.0f);
    for(int i = 0; i < size; i++) {
        for(int j = 0; j < nFeatures; j++) {
            input[i*nFeatures + j] = (float) inputs[i][j] / 255.0;
        }
    }

    //read in output file
    std::vector<std::vector<int>> outputs = readDataFromUByteFile(train_label_path);
    std::vector<std::vector<int>> test_outputs = readDataFromUByteFile(test_label_path);

    std::vector<float> weights = initializeRandomArray(nFeatures, numClasses);

    float * product = (float*)malloc(BATCH_SIZE*numClasses*sizeof(float));
    float * output;
    output = (float*)malloc(sizeof(float) * size * numClasses);
    for (int i = 0; i < size; i++) {
        for(int j = 0; j < numClasses; j++) {
            output[i*numClasses+j] = 0;
        }
        output[i*numClasses+outputs[i][0]] = 1.0;
    }
    float *d_outputs = transferMatrixToDevice(output, size, 10);

    //declare device variables
    float *d_inputs;
    float *d_weights;
    float *d_product;
    float *d_gradients;
    float *d_test_inputs;
    cudaMalloc(&d_gradients, nThreadsPerWorker*nWorkers*nFeatures*numClasses*sizeof(float));
    cudaMalloc(&d_inputs, size*nFeatures*sizeof(float));
    cudaMemcpy(d_inputs, input.data(), size*nFeatures*sizeof(float), cudaMemcpyHostToDevice);
    cudaMalloc(&d_weights, nFeatures*numClasses*sizeof(float));
    cudaMemcpy(d_weights, weights.data(), nFeatures*numClasses*sizeof(float), cudaMemcpyHostToDevice);
    cudaMalloc(&d_product, BATCH_SIZE*numClasses*sizeof(float));
    cudaMemcpy(d_product, product, BATCH_SIZE*numClasses*sizeof(float), cudaMemcpyHostToDevice);

    //declare variables for computing metrics
    int correct = 0;
    float logLoss = 0.0;
    float accuracy = 0.0;
    auto endInitial = std::chrono::system_clock::now();
    // std::chrono::duration<double> elapsed_seconds = endInitial-startInitial;
    // std::cout << "Finished initial setup in " << elapsed_seconds.count() << " seconds" << std::endl;
    auto startTrain = std::chrono::system_clock::now();
    for(int i = 0; i < nEpochs; i++) {
        correct = 0;
        logLoss = 0;
        accuracy = 0.0;
        for(int j = 0; j < size; j += BATCH_SIZE) {
            //forward pass
            // std::cout << "Starting forward pass..." << std::endl;
            // auto initForward = std::chrono::system_clock::now();
            predict<<<nWorkers, nThreadsPerWorker>>>(d_inputs+(j*nFeatures), d_weights, d_product, BATCH_SIZE, nFeatures, numClasses);
            cudaDeviceSynchronize();
            cudaMemcpy(product, d_product, BATCH_SIZE*numClasses*sizeof(float), cudaMemcpyDeviceToHost);
            // printf("Probabilities\n");
            // printMatrix(product, BATCH_SIZE, numClasses);
            correct += getAccuracy(product, outputs, BATCH_SIZE, numClasses, j);
            logLoss += crossEntropyLoss(product, outputs, BATCH_SIZE, numClasses, j);
            accuracy = (correct) / (float) (BATCH_SIZE);
            // printf("Accuracy: %f%%\n", accuracy*100);
            // printf("Log loss: %f\n", logLoss);
            forward_pass<<<nWorkers, nThreadsPerWorker>>>(d_inputs+(j*nFeatures), d_weights, d_outputs+(j*nFeatures), d_product, d_gradients, BATCH_SIZE, nFeatures, numClasses);
            cudaDeviceSynchronize();

            // auto endForward = std::chrono::system_clock::now();
            // std::chrono::duration<double> elapsed_forward = endForward - initForward;
            // std::cout << "Finished forward pass in " << elapsed_forward.count() << " seconds" << std::endl;
            // //backward pass
            // std::cout << "Starting backward pass..." << std::endl;
            auto initBackward = std::chrono::system_clock::now();
            ringReduce<<<nWorkers, nThreadsPerWorker>>>(d_gradients, nThreadsPerWorker*nWorkers, nFeatures*numClasses, nFeatures*numClasses/(nThreadsPerWorker*nWorkers));
            cudaDeviceSynchronize();
            // float * gradients = (float*)malloc(nFeatures*numClasses*sizeof(float));
            // printf("Weights\n");
            backward_pass<<<nWorkers, nThreadsPerWorker>>>(d_weights, d_gradients, BATCH_SIZE, learning_rate, nFeatures, numClasses);
            // cudaMemcpy(gradients, d_weights, nFeatures*numClasses*sizeof(float), cudaMemcpyDeviceToHost);
            // printMatrix(gradients, nFeatures, numClasses);
            // printf("Weights\n");
            cudaDeviceSynchronize();

            // auto endBackward = std::chrono::system_clock::now();
            // std::chrono::duration<double> elapsed_backward = endBackward - initBackward;
            // std::cout << "Finished backward pass in " << elapsed_backward.count() << " seconds" << std::endl;
        }
        accuracy = (correct) / (float) (size);
        printf("End of epoch %d\n", i+1);
        printf("Accuracy: %f%%\n", accuracy*100);
        printf("Log loss: %f\n", logLoss);
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