#include "../include/log_reg.h"
#include "../include/lin_alg.h"
#include "../include/utils.h"
#include <iostream>
#include <chrono> 

__global__ void ringReduce(LogisticRegression * model, const int total_steps, const int step_size, const int chunk_size) {
    int index = blockIdx.x*blockDim.x + threadIdx.x;
    int begin_part = index*chunk_size;
    int end_part = (index+1)*chunk_size;
    for(int i = 1; i < total_steps; i++) {
        for(int j = begin_part; j < end_part; j++) {
            model->gradients[j] += model->gradients[(i*step_size)+j];
        }
    }
    // printf("Ring reduce\n");
}

// __global__ void predict(LogisticRegression* model, float* inputs, float* product, int size) {
//     int i = blockIdx.x*blockDim.x + threadIdx.x;
//     int batch = size / (blockDim.x * gridDim.x);
//     dotProduct(inputs+(i*(model->nFeatures)*batch), model->weights, product+(i*(model->nClasses)*batch), batch, model->nFeatures, model->nFeatures, model->nClasses);
//     softmax(product+(i*(model->nClasses)*batch), batch, (model->nClasses));
// }

// __global__ void forward_pass(LogisticRegression* model, float* inputs, float* outputs, float* product, int size, int nClasses) {
//     int i = blockIdx.x*blockDim.x + threadIdx.x;
//     int batch = size / (blockDim.x * gridDim.x);
//     matrixSubtract(product+(i*nClasses*batch), outputs+(i*nClasses*batch), batch, nClasses, batch, nClasses, product+(i*nClasses*batch));
//     dotProductTranspose(inputs+(i*batch*(model->nFeatures)), product+(i*batch*(model->nClasses)), ((*model).gradients)+(i*(model->nClasses)*(model->nFeatures)), batch, (model->nFeatures), batch, (model->nClasses));
// }

__global__ void backward_pass(LogisticRegression* model, int batch_size, float learning_rate) {
    int index = blockIdx.x*blockDim.x + threadIdx.x;
    //BATCH_SIZE*n_classes length vector
    int batch = model->nFeatures / (blockDim.x * gridDim.x);
    int start = index*batch*(model->nClasses);
    for(int i = 0; i < batch; i++) {
        for(int j = 0; j < model->nClasses; j++) {
            (*model).gradients[start+i*(model->nClasses)+j] *= (learning_rate / batch_size);
            (*model).weights[start+i*(model->nClasses)+j] -=  (*model).gradients[start+i*(model->nClasses)+j];
        }
    }
    // printf("Finish backward\n");
}

LogisticRegression* copyModelToGPU(LogisticRegression *model, int nWorkers, int nThreadsPerWorker) {
    //define pointer for GPU's copy of the model
    LogisticRegression* d_model;
    //allocate space in the GPU memory for the model
    cudaMalloc(&d_model, sizeof(LogisticRegression));

    //declare all of the pointers that we need on the GPU (weights and gradients) and pass them to device
    float *d_weights;
    cudaMalloc(&d_weights, model->nFeatures*model->nClasses*sizeof(float));
    cudaMemcpy(d_weights, (*model).weights, model->nFeatures*model->nClasses*sizeof(float), cudaMemcpyHostToDevice);
    float *d_gradients;
    cudaMalloc(&d_gradients, nThreadsPerWorker*nWorkers*model->nFeatures*model->nClasses*sizeof(float));
    cudaMemcpy(d_gradients, (*model).gradients, nThreadsPerWorker*nWorkers*model->nFeatures*model->nClasses*sizeof(float), cudaMemcpyHostToDevice);
    //create temp model
    LogisticRegression temp = *model;
    temp.weights = d_weights;
    temp.gradients = d_gradients;
    temp.nFeatures = model->nFeatures;
    temp.nClasses = model->nClasses;
    //pass temp model to GPU
    cudaMemcpy(d_model, &temp, sizeof(LogisticRegression), cudaMemcpyHostToDevice);
    return d_model;
}

void train(LogisticRegression *model, float* train_input, std::vector<std::vector<int>>& train_labels, float* test_input, std::vector<std::vector<int>>& test_labels, 
    int nEpochs, int batch_size, int total_size, int test_size, float learning_rate, int nWorkers, int nThreadsPerWorker) {
    //since we can't directly access device variables from the host function, we'll have to do everything here
    LogisticRegression *d_model = copyModelToGPU(model, nWorkers, nThreadsPerWorker);
    std::cout << "TEST SIZE " << test_size << std::endl;
    //copy train data
    float *d_inputs;
    //copy weights
    cudaMalloc(&d_inputs, total_size*(model->nFeatures)*sizeof(float));
    cudaMemcpy(d_inputs, train_input, total_size*(model->nFeatures)*sizeof(float), cudaMemcpyHostToDevice);

    //copy test data
    float *d_test_inputs;
    cudaMalloc(&d_test_inputs, test_size*(model->nFeatures)*sizeof(float));
    cudaMemcpy(d_test_inputs, test_input, test_size*(model->nFeatures)*sizeof(float), cudaMemcpyHostToDevice);

    //convert labels to one hot encoding
    float * one_hot = (float *)malloc(sizeof(float) * total_size * model->nClasses);
    for (int i = 0; i < total_size; i++) {
        for(int j = 0; j < model->nClasses; j++) {
            one_hot[i*model->nClasses+j] = 0;
        }
        one_hot[i*(model->nClasses)+train_labels[i][0]] = 1.0;
    }
    //pass labels to GPU
    float *d_outputs = transferMatrixToDevice(one_hot, total_size, model->nClasses);

    //initialize array for storing predictions on host
    float * predictions = (float*)malloc(batch_size*(model->nClasses)*sizeof(float));
    float * d_product = transferMatrixToDevice(predictions, batch_size, model->nClasses);
    //initialize array for storing predictions of test set on host
    float * test_predictions = (float*)malloc(test_size*model->nClasses*sizeof(float));
    float * d_test_product = transferMatrixToDevice(test_predictions, test_size, model->nClasses);
    //define metrics
    int correct = 0;
    double logLoss = 0.0;
    float accuracy = 0.0;
    for(int i = 0; i < nEpochs; i++) {
        correct = 0;
        logLoss = 0;
        accuracy = 0.0;

        for(int j = 0; j < total_size; j+=batch_size) {

            // auto endForward = std::chrono::system_clock::now();
            // std::chrono::duration<double> elapsed_forward = endForward - initForward;
            // std::cout << "Finished forward pass in " << elapsed_forward.count() << " seconds" << std::endl;
            // //backward pas

        }
        accuracy = correct / (float) total_size;
        printf("Accuracy: %f%%\n", accuracy*100);
        printf("Log loss: %f\n", logLoss);
        cudaDeviceSynchronize();
        std::cout << "Finished eval" << std::endl;
        cudaMemcpy(test_predictions, d_test_product, test_size*(model->nClasses)*sizeof(float), cudaMemcpyDeviceToHost);
        int test_correct = getAccuracy(test_predictions, test_labels, test_size, model->nClasses, 0);
        double test_loss = crossEntropyLoss(test_predictions, test_labels, test_size, model->nClasses, 0);
        printf("Test log loss: %f\nTest accuracy %f%%\n", test_loss, test_correct / (float) test_size * 100);
    }
    cudaFree(d_model);
    cudaFree(d_inputs);
    cudaFree(d_test_inputs);
    cudaFree(d_outputs);
    cudaFree(d_product);
    cudaFree(d_test_product);
}
