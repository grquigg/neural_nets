#include <iostream>
#include <cassert>
#include <string>
#include <vector>
// #include "utils.h"
#include "../include/utils.h"
#include "../include/lin_alg.h"

float* transferMatrixToDevice(float *matrix, int height, int width) {
    float* deviceMatrix;
    cudaMalloc(&deviceMatrix, height*width*sizeof(float));
    for(int i = 0; i < height; i++) {
        cudaMemcpy(deviceMatrix+(i*width), matrix+(i*width), sizeof(float)*width, cudaMemcpyHostToDevice);
    }
    return deviceMatrix;
}


int main() {
    int test_size = 40;
    int n_features = 20;
    int out_classes = 5;
    int nWorkers = 1;
    int nThreadsPerWorker = 1;
    float learning_rate = 0.001;
    int BATCH_SIZE = 20;
    int epochs = 100;
    float * inputs = (float *)malloc(n_features*test_size*sizeof(float));
    for(int i = 0; i < test_size; i++) {
        for(int j = 0; j < n_features; j++) {
            inputs[(i*n_features)+j] = 0;
        }
        inputs[(i*n_features) + (i % n_features)] = 1.0;
    }
    printMatrix(inputs, test_size, n_features);
    float * outputs = (float*)malloc(test_size*out_classes*sizeof(float));
    for(int i = 0; i < test_size; i++) {
        for(int j = 0; j < out_classes; j++) {
            outputs[i*out_classes+j] = 0.0;
        }
        outputs[(i*out_classes) + (i % out_classes)] = 1.0;
    }

    float * matrix = initializeFlatRandomArray(n_features, out_classes);
    float *d_gradients;
    float *d_inputs;
    float *d_weights;
    float *d_product;
    cudaMalloc(&d_gradients, nWorkers*nThreadsPerWorker*n_features*out_classes*sizeof(float));
    cudaMalloc(&d_inputs, test_size*n_features*sizeof(float));
    cudaMemcpy(d_inputs, inputs, test_size*n_features*sizeof(float), cudaMemcpyHostToDevice);
    cudaMalloc(&d_weights, n_features*out_classes*sizeof(float));
    cudaMemcpy(d_weights, matrix, n_features*out_classes*sizeof(float), cudaMemcpyHostToDevice);
    cudaMalloc(&d_product, BATCH_SIZE*out_classes*sizeof(float));
    float *d_outputs = transferMatrixToDevice(outputs, test_size, out_classes);
    int *correct = (int*)malloc(sizeof(int));
    float *logLoss = (float*)malloc(sizeof(float));
    correct[0] = 0;
    logLoss[0] = 0;
    for(int j = 0; j < epochs; j++) {
        correct[0] = 0;
        logLoss[0] = 0;
        for(int i = 0; i < test_size; i+=BATCH_SIZE) {
            predict<<<nWorkers, nThreadsPerWorker>>>(d_inputs+(i*n_features), d_weights, d_product, BATCH_SIZE, n_features, out_classes);
            cudaDeviceSynchronize();
            float * product = (float*)malloc(BATCH_SIZE*out_classes*sizeof(float));
            cudaMemcpy(product, d_product, BATCH_SIZE*out_classes*sizeof(float), cudaMemcpyDeviceToHost);
            printf("Call from host %d\n", correct[0]);
            printMatrix(product, BATCH_SIZE, out_classes);
            printf("Actual\n");
            printMatrix(outputs+(i*out_classes), BATCH_SIZE, out_classes);
            correct[0] += getAccuracy(product, outputs+(i*out_classes), BATCH_SIZE, out_classes);
            logLoss[0] += crossEntropyLoss(product, outputs+(i*out_classes), BATCH_SIZE, out_classes);

            forward_pass<<<nWorkers, nThreadsPerWorker>>>(d_inputs+(i*n_features), d_weights, d_outputs+(i*out_classes), d_product+(i*out_classes), d_gradients, BATCH_SIZE, n_features, out_classes);
            cudaDeviceSynchronize();
            // printf("Outputs\n");
            // printMatrix(outputs+(i*out_classes), BATCH_SIZE, out_classes);
            float * gradients = (float*)malloc(nWorkers*nThreadsPerWorker*n_features*out_classes*sizeof(float));
            cudaMemcpy(gradients, d_gradients, nWorkers*nThreadsPerWorker*n_features*out_classes*sizeof(float), cudaMemcpyDeviceToHost);
            // printMatrix(gradients, nWorkers*nThreadsPerWorker*n_features, out_classes);
            //aggregate gradients across different shards
            ringReduce<<<nWorkers, nThreadsPerWorker>>>(d_gradients, nThreadsPerWorker*nWorkers, n_features*out_classes, (n_features*out_classes)/(nThreadsPerWorker*nWorkers));
            cudaMemcpy(gradients, d_gradients, nWorkers*nThreadsPerWorker*n_features*out_classes*sizeof(float), cudaMemcpyDeviceToHost);
            // printf("Gradients\n");
            // printMatrix(gradients, n_features, out_classes);
            //backward pass
            backward_pass<<<nWorkers, nThreadsPerWorker>>>(d_weights, d_gradients, BATCH_SIZE, learning_rate, n_features, out_classes);
            cudaMemcpy(matrix, d_weights, n_features*out_classes*sizeof(float), cudaMemcpyDeviceToHost);
            printf("Matrix\n");
            printMatrix(matrix, n_features, out_classes);

        }
        float accuracy = correct[0] / (float)(test_size);
        std::cout << "Accuracy: "<< accuracy *100 << "%" << std::endl;
        std::cout << "Loss: "<< logLoss[0] << std::endl;
    }
    return 0;
}

