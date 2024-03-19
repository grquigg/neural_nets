#ifndef NN_H
#define NN_H
#include <vector>
#include <iostream>
#include <cublas_v2.h>

struct NeuralNetwork {
    int nClasses;
    int nLayers;
    int * layer_size;
    float ** weights;
    float ** biases;
    float lambda;
    float ** gradients;
    float ** grad_biases;
};

NeuralNetwork * copyModelToGPU(NeuralNetwork *model, int nWorkers, int nThreadsPerWorker);

/*
The order of the arguments that should be passed into the train function are as follows:
1. Reference to the HOST model struct
2. Reference to the train input array
3. Reference to the train labels array
4. Reference to the test input array
5. Reference to the test labels array
////HYPERPARAMS
6. Number of epochs
7. Batch size
8. Full size of the entire train set
9. learning rate
10. number of desired workers
11. number of threads per worker
*/

void train(NeuralNetwork *model, float* train_input, std::vector<std::vector<int>>& train_labels, float* test_input, std::vector<std::vector<int>>& test_labels, 
int nEpochs, int batch_size, int total_size, int test_size, float learning_rate, int nWorkers, int nThreadsPerWorker);

__global__ void predict(NeuralNetwork* model, float* inputs, float* activations,  int* offsets, int size, cublasHandle_t handle);

__global__ void ringReduce(NeuralNetwork* model, const int total_steps);

__global__ void backward_pass(NeuralNetwork* model, int batch_size, float learning_rate);

__global__ void backprop(NeuralNetwork* model, float* inputs, float* outputs, float* activations, float* deltas, int* offsets, int size, int nClasses);
////DEBUGGING FUNCTIONS
__global__ void auditGradients(NeuralNetwork* model);

__global__ void auditDeltas(NeuralNetwork* model, float* deltas, int* offsets, int batches, int batch_size);

__global__ void auditWeights(NeuralNetwork* model);

#endif