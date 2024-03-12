#include "../include/utils.h"
#include "../include/lin_alg.h"
#include "../include/models.h"
#include <chrono> 
#include <iostream>

float* transferMatrixToDevice(float *matrix, int height, int width) {
    float* deviceMatrix;
    cudaMalloc(&deviceMatrix, height*width*sizeof(float));
    for(int i = 0; i < height; i++) {
        cudaMemcpy(deviceMatrix+(i*width), matrix+(i*width), sizeof(float)*width, cudaMemcpyHostToDevice);
    }
    return deviceMatrix;
}

// LogisticRegression * copyModelToHost(LogisticRegression *model, LogisticRegression *start) {
//     LogisticRegression* host;
//     host->weights = initializeFlatRandomArray(nFeatures, numClasses);
//     host->gradients = (float*)malloc(nFeatures*numClasses*sizeof(float));
// }
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

NeuralNetwork * copyModelToGPU(NeuralNetwork *model, int nWorkers, int nThreadsPerWorker) {
    NeuralNetwork* d_model;
    int * nLayers;
    float **d_weights;
    float **d_weights_t;
    float **d_biases;
    float **d_gradients;
    float **d_grad_biases;
    //allocate all of the memory that we need to CUDA
    cudaMalloc(&d_model, sizeof(NeuralNetwork));
    cudaMalloc(&nLayers, (model->nLayers+1)*sizeof(int));
    cudaMemcpy(nLayers, model->layer_size, (model->nLayers+1)*sizeof(int), cudaMemcpyHostToDevice);
    // // cudaMalloc(&d_weights, (model->nLayers)*sizeof(float*));
    // cudaMalloc(&d_biases, (model->nLayers)*sizeof(float*));
    float **temp_weights = new float*[model->nLayers];
    float **temp_weights_t = new float*[model->nLayers];
    float **temp_biases = new float*[model->nLayers];
    float **temp_gradients = new float*[model->nLayers];
    float **temp_grad_biases = new float*[model->nLayers];
    for(int i = 1; i < model->nLayers+1; i++) {
        cudaMalloc(&temp_weights[i-1], model->layer_size[i-1]*model->layer_size[i]*sizeof(float));
        cudaMemcpy(temp_weights[i-1], model->weights[i-1], model->layer_size[i-1]*model->layer_size[i]*sizeof(float), cudaMemcpyHostToDevice);
        cudaMalloc(&temp_biases[i-1], model->layer_size[i]*sizeof(float));
        cudaMemcpy(temp_biases[i-1], model->biases[i-1], model->layer_size[i]*sizeof(float), cudaMemcpyHostToDevice);
        cudaMalloc(&temp_gradients[i-1], nThreadsPerWorker*nWorkers*model->layer_size[i-1]*model->layer_size[i]*sizeof(float));
        cudaMalloc(&temp_weights_t[i-1], model->layer_size[i-1]*model->layer_size[i]*sizeof(float));
        // cudaMemcpy(temp_gradients[i-1], model->gradients[i-1], nThreadsPerWorker*nWorkers*model->layer_size[i-1]*model->layer_size[i]*sizeof(float), cudaMemcpyHostToDevice);
        cudaMalloc(&temp_grad_biases[i-1],  nThreadsPerWorker*nWorkers*model->layer_size[i]*sizeof(float));
    }
    cudaMalloc(&d_gradients, (model->nLayers)*sizeof(float*));
    cudaMemcpy(d_gradients, temp_gradients, (model->nLayers)*sizeof(float*), cudaMemcpyHostToDevice);
    cudaMalloc(&d_grad_biases, (model->nLayers)*sizeof(float*));
    cudaMemcpy(d_grad_biases, temp_grad_biases, (model->nLayers)*sizeof(float*), cudaMemcpyHostToDevice);
    cudaMalloc(&d_biases, (model->nLayers)*sizeof(float*));
    cudaMemcpy(d_biases, temp_biases, (model->nLayers)*sizeof(float*), cudaMemcpyHostToDevice);
    cudaMalloc(&d_weights, (model->nLayers)*sizeof(float*));
    cudaMemcpy(d_weights, temp_weights, (model->nLayers)*sizeof(float*), cudaMemcpyHostToDevice);
    cudaMalloc(&d_weights_t, (model->nLayers)*sizeof(float*));
    cudaMemcpy(d_weights_t, temp_weights_t, (model->nLayers)*sizeof(float*), cudaMemcpyHostToDevice);
    NeuralNetwork temp = *model;
    temp.nClasses = model->nClasses;
    temp.nLayers = model->nLayers;
    temp.layer_size = nLayers;
    temp.weights = d_weights;
    temp.weight_transpose = d_weights_t;
    temp.gradients = d_gradients;
    temp.biases = d_biases;
    temp.grad_biases = d_grad_biases;
    temp.lambda = model->lambda;
    cudaMemcpy(d_model, &temp, sizeof(NeuralNetwork), cudaMemcpyHostToDevice);
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
            predict<<<nWorkers, nThreadsPerWorker>>>(d_model, d_inputs+(j*model->nFeatures), d_product, batch_size);
            // predict<<<nWorkers, nThreadsPerWorker>>>(d_inputs+(j*model->nFeatures), d_model->weights, d_product, batch_size, model->nFeatures, model->nClasses);
            cudaDeviceSynchronize();
            cudaMemcpy(predictions, d_product, batch_size*(model->nClasses)*sizeof(float), cudaMemcpyDeviceToHost);
            // printf("Probabilities\n");
            // printMatrix(product, batch_size, model->nClasses);
            // correct += getAccuracy(predictions, train_labels, batch_size, model->nClasses, j);
            // logLoss += crossEntropyLoss(predictions, train_labels, batch_size, model->nClasses, j);
            forward_pass<<<nWorkers, nThreadsPerWorker>>>(d_model, d_inputs+(j*(model->nFeatures)), d_outputs+(j*(model->nClasses)), d_product, batch_size, model->nClasses);
            cudaDeviceSynchronize();

            // auto endForward = std::chrono::system_clock::now();
            // std::chrono::duration<double> elapsed_forward = endForward - initForward;
            // std::cout << "Finished forward pass in " << elapsed_forward.count() << " seconds" << std::endl;
            // //backward pass
            // std::cout << "Starting backward pass..." << std::endl;
            auto initBackward = std::chrono::system_clock::now();
            ringReduce<<<nWorkers, nThreadsPerWorker>>>(d_model, nThreadsPerWorker*nWorkers, model->nFeatures*model->nClasses, model->nFeatures*model->nClasses/(nThreadsPerWorker*nWorkers));
            cudaDeviceSynchronize();
            // float * gradients = (float*)malloc(nFeatures*model->nClasses*sizeof(float));
            // printf("Weights\n");
            // printMatrix(model->weights, model->nFeatures, model->nClasses);
            backward_pass<<<nWorkers, nThreadsPerWorker>>>(d_model, batch_size, learning_rate);
            cudaDeviceSynchronize();

        }
        accuracy = correct / (float) total_size;
        printf("Accuracy: %f%%\n", accuracy*100);
        printf("Log loss: %f\n", logLoss);
        predict<<<10, 10>>>(d_model, d_test_inputs, d_test_product, test_size);
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

void train(NeuralNetwork *model, float* train_input, std::vector<std::vector<int>>& train_labels, float* test_input, std::vector<std::vector<int>>& test_labels, 
int nEpochs, int batch_size, int total_size, int test_size, float learning_rate, int nWorkers, int nThreadsPerWorker) {
    printf("Train network\n");
    NeuralNetwork *d_model = copyModelToGPU(model, nWorkers, nThreadsPerWorker);
    std::cout << "TEST SIZE " << test_size << std::endl;
    //copy train data
    float *d_inputs;
    //copy weights
    cudaError_t error;
    error = cudaMalloc(&d_inputs, total_size*(model->layer_size[0])*sizeof(float));
    if(error != cudaSuccess) {
        std::cout << "Problem with copying" << std::endl;
    }
    error = cudaMemcpy(d_inputs, train_input, total_size*(model->layer_size[0])*sizeof(float), cudaMemcpyHostToDevice);
    if(error != cudaSuccess) {
        std::cout << "Problem" << std::endl;
    }
    //copy test data
    float *d_test_inputs;
    cudaMalloc(&d_test_inputs, test_size*(model->layer_size[0])*sizeof(float));
    cudaMemcpy(d_test_inputs, test_input, test_size*(model->layer_size[0])*sizeof(float), cudaMemcpyHostToDevice);

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

    //initialize array for storing intermediary activation functions on GPU
    /*the super nice thing about the parallelized computation of neural networks is 
    ALL of the activation functions take the form of (BATCH_SIZE, layer_size)
    Which means we can likely have all of the activations stored via one pointer and only
    have to allocate the memory ONCE. However, since I have absolutely no idea what I'm doing,
    I'm gonna stay away from that for now.

    Since double pointers don't want to cooperate for some reason, and since it doesn't make sense
    for this huge block of memory to be allocated several different times randomly in memory, we allocate a single block
    of memory as well as an integer array to keep track of the offsets of each "block" in memory.
    */
    int activations_size = 0;
    int * offsets = new int[model->nLayers];
    for(int i = 1; i <= model->nLayers; i++) {
        offsets[i-1] = (batch_size * activations_size);
        activations_size += model->layer_size[i];
        printf("Offset at %d %d\n", i-1, offsets[i-1]);
    }
    float * d_activations = new float[batch_size*activations_size];
    float * activations = new float[batch_size*activations_size];
    printf("Activations size: %d\n", batch_size*activations_size);
    //device pointers
    int * d_offsets;
    cudaMalloc(&d_activations, activations_size*batch_size*sizeof(float));
    cudaMalloc(&d_offsets, model->nLayers*sizeof(int));
    for(int i = 0; i < activations_size*batch_size; i++) {
        activations[i] = 1;
    }
    cudaMemcpy(d_activations, activations, activations_size*batch_size*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_offsets, offsets, model->nLayers*sizeof(int), cudaMemcpyHostToDevice);

    //deltas
    float * d_deltas = new float[batch_size*activations_size];
    cudaMalloc(&d_deltas, activations_size*batch_size*sizeof(float));
    cudaMemcpy(d_activations, activations, activations_size*batch_size*sizeof(float), cudaMemcpyHostToDevice);
    // float * d_product = transferMatrixToDevice(activations, batch_size, activations_size);
    // //initialize array for storing predictions of test set on host
    float * test_predictions = (float*)malloc(test_size*model->nClasses*sizeof(float));
    float * d_test_product = transferMatrixToDevice(test_predictions, test_size, model->nClasses);
    //define metrics
    int correct = 0;
    double logLoss = 0.0;
    float accuracy = 0.0;
    auto startTrain = std::chrono::system_clock::now();
    for(int i = 0; i < nEpochs; i++) {
        correct = 0;
        logLoss = 0;
        accuracy = 0.0;

        for(int j = 0; j < 1000; j+=batch_size) {
            //pass inputs through the network
            setTranspose<<<1,1>>>(d_model);
            cudaDeviceSynchronize();
            auto startForward = std::chrono::system_clock::now();
            predict<<<nWorkers, nThreadsPerWorker>>>(d_model, d_inputs+(j*model->layer_size[0]), d_activations, d_offsets, batch_size);
            cudaDeviceSynchronize();
            auto endForward = std::chrono::system_clock::now();
            std::chrono::duration<double> elapsed_forward = endForward - startForward;
            std::cout << "Finished forward pass in " << elapsed_forward.count() << " seconds" << std::endl;
            float* predictions = (float*)malloc(activations_size*batch_size*sizeof(float));
            error = cudaMemcpy(predictions, d_activations, activations_size*batch_size*sizeof(float), cudaMemcpyDeviceToHost);
            // for(int k = 0; k < model->nLayers; k++) {
            //     printf("Activations at layer %d\n", k);
            //     printMatrix(predictions+offsets[k], batch_size, model->layer_size[k+1]);
            // }
            correct += getAccuracy(predictions+offsets[1], train_labels, batch_size, model->nClasses, j);
            logLoss += crossEntropyLoss(predictions+offsets[1], train_labels, batch_size, model->nClasses, j);
            // printf("Accuracy: %f%%\n", correct / (float) batch_size * 100);
            // printf("Log loss %f\n", logLoss);
            // //compute gradients in forward_pass
            auto startBackward = std::chrono::system_clock::now();
            backprop<<<nWorkers, nThreadsPerWorker>>>(d_model, d_inputs+(j*(model->layer_size[0])), d_outputs+(j*(model->nClasses)), d_activations, d_deltas, d_offsets, batch_size, model->nClasses);
            cudaDeviceSynchronize();
            auto endBackward = std::chrono::system_clock::now();
            std::chrono::duration<double> elapsed_backward = endBackward - startBackward;
            std::cout << "Finished backward pass in " << elapsed_backward.count() << " seconds" << std::endl;
            auto startReduce = std::chrono::system_clock::now();
            ringReduce<<<nWorkers, nThreadsPerWorker>>>(d_model, nWorkers*nThreadsPerWorker);
            cudaDeviceSynchronize();
            auto endReduce = std::chrono::system_clock::now();
            std::chrono::duration<double> elapsed_reduce = endReduce - startReduce;
            std::cout << "Finished ring reduce in " << elapsed_reduce.count() << " seconds" << std::endl;
            // auditDeltas<<<1,1>>>(d_model, d_deltas, d_offsets, nWorkers*nThreadsPerWorker, batch_size);
            // cudaDeviceSynchronize();
            // auditGradients<<<1,1>>>(d_model);
            // cudaDeviceSynchronize();
            auto startUpdate = std::chrono::system_clock::now();
            backward_pass<<<nWorkers, nThreadsPerWorker>>>(d_model, batch_size, learning_rate);
            cudaDeviceSynchronize();
            auto endUpdate = std::chrono::system_clock::now();
            std::chrono::duration<double> elapsed_update = endUpdate - startUpdate;
            std::cout << "Finished weight update in " << elapsed_update.count() << " seconds" << std::endl;
        }
        accuracy = correct / (float) total_size;
        printf("End of epoch %d\n", i+1);
        printf("Accuracy: %f%%\n", accuracy*100);
        printf("Log loss: %f\n", logLoss);
    }
    auto endTrain = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed_forward = endTrain - startTrain;
    std::cout << "Finished forward pass in " << elapsed_forward.count() << " seconds" << std::endl;
    cudaFree(d_model);
    cudaFree(d_inputs);
    cudaFree(d_test_inputs);
    cudaFree(d_outputs);
    cudaFree(d_activations);
    cudaFree(d_test_product);
    cudaFree(d_deltas);
}