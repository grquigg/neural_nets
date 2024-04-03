#include <gtest/gtest.h>
#include "models.h"
#include "lin_alg.h"
#include "example2.h"

TEST_F(Example2Suite, TestForwardPass_Thread1_BatchSize1) {
    int nWorkers = 1;
    int nThreadsPerWorker = 1;
    dim3 nBlocks(nWorkers, 1, 1);
    dim3 nThreads(nThreadsPerWorker, 1, 1);
    int batch_size = 1;
    model->setupGPU(nThreadsPerWorker*nWorkers, batch_size);
    float *d_product;
    cudaMalloc(&d_inputs, n_inputs*model->layer_size[0]*sizeof(float));
    cudaMemcpy(d_inputs, input, n_inputs*model->layer_size[0]*sizeof(float), cudaMemcpyHostToDevice);
    cudaMalloc(&d_product, n_inputs*model->layer_size[1]*sizeof(float));
    for(int i = 0; i < n_inputs; i += batch_size) {
        dotProductSegmented<<<nBlocks, nThreads>>>(d_inputs+(i*model->layer_size[0]), model->d_weights[0], d_product+(i*model->layer_size[1]), batch_size, model->layer_size[0], model->layer_size[0], model->layer_size[1], model->d_biases[0]);
        cudaDeviceSynchronize();
    }
    float *prod = new float[n_inputs*model->layer_size[1]];
    for(int i = 0; i < n_inputs*model->layer_size[1]; i++) {
        prod[i] = 1.0f;
    }
    cudaMemcpy(prod, d_product, n_inputs*model->layer_size[1]*sizeof(float), cudaMemcpyDeviceToHost);
    for(int i = 0; i < n_inputs*model->layer_size[1]; i++) {
        EXPECT_FLOAT_EQ(prod[i], correctForward[i]);
    }
}

TEST_F(Example2Suite, TestForwardPass_Thread2_BatchSize1) {
    int nWorkers = 1;
    int nThreadsPerWorker = 1;
    int batch_size = 1;
    model->setupGPU(nThreadsPerWorker*nWorkers, batch_size);
    float *d_product;
    cudaMalloc(&d_inputs, n_inputs*model->layer_size[0]*sizeof(float));
    cudaMemcpy(d_inputs, input, n_inputs*model->layer_size[0]*sizeof(float), cudaMemcpyHostToDevice);
    cudaMalloc(&d_product, n_inputs*model->layer_size[1]*sizeof(float));
    dim3 nBlocks(nWorkers, batch_size, 1);
    dim3 nThreads(nThreadsPerWorker, model->layer_size[0], 1);
    for(int i = 0; i < n_inputs; i += batch_size) {
        dotProductSegmented<<<nBlocks, nThreads>>>(d_inputs+(i*model->layer_size[0]), model->d_weights[0], d_product+(i*model->layer_size[1]), batch_size, model->layer_size[0], model->layer_size[0], model->layer_size[1], model->d_biases[0]);
        cudaDeviceSynchronize();
    }
    float *prod = new float[n_inputs*model->layer_size[1]];
    for(int i = 0; i < n_inputs*model->layer_size[1]; i++) {
        prod[i] = 1.0f;
    }
    cudaMemcpy(prod, d_product, n_inputs*model->layer_size[1]*sizeof(float), cudaMemcpyDeviceToHost);
    for(int i = 0; i < n_inputs*model->layer_size[1]; i++) {
        EXPECT_FLOAT_EQ(prod[i], correctForward[i]);
    }
}

TEST_F(Example2Suite, TestForwardPass_Thread1_BatchSize2) {
    int nWorkers = 1;
    int nThreadsPerWorker = 1;
    dim3 nBlocks(nWorkers, 1, 1);
    dim3 nThreads(nThreadsPerWorker, 1, 1);
    int batch_size = 2;
    model->setupGPU(nThreadsPerWorker*nWorkers, batch_size);
    float *d_product;
    cudaMalloc(&d_inputs, n_inputs*model->layer_size[0]*sizeof(float));
    cudaMemcpy(d_inputs, input, n_inputs*model->layer_size[0]*sizeof(float), cudaMemcpyHostToDevice);
    cudaMalloc(&d_product, batch_size*model->layer_size[1]*sizeof(float));
    dotProductSegmented<<<nBlocks, nThreads>>>(d_inputs, model->d_weights[0], d_product, batch_size, model->layer_size[0], model->layer_size[0], model->layer_size[1], model->d_biases[0]);
    cudaDeviceSynchronize();
    float *prod = new float[batch_size*model->layer_size[1]];
    for(int i = 0; i < n_inputs*model->layer_size[1]; i++) {
        prod[i] = 1.0f;
    }
    cudaMemcpy(prod, d_product, batch_size*model->layer_size[1]*sizeof(float), cudaMemcpyDeviceToHost);
    for(int i = 0; i < 8; i++) {
        EXPECT_FLOAT_EQ(prod[i], correctForward[i]);
    }
}

TEST_F(Example2Suite, TestForwardPass_Thread2_BatchSize2) {
    int nWorkers = 2;
    int nThreadsPerWorker = 1;
    dim3 nBlocks(nWorkers, 1, 1);
    dim3 nThreads(nThreadsPerWorker, 1, 1);
    int batch_size = 2;
    model->setupGPU(nThreadsPerWorker*nWorkers, batch_size);
    float *d_product;
    cudaMalloc(&d_inputs, n_inputs*model->layer_size[0]*sizeof(float));
    cudaMemcpy(d_inputs, input, n_inputs*model->layer_size[0]*sizeof(float), cudaMemcpyHostToDevice);
    cudaMalloc(&d_product, batch_size*model->layer_size[1]*sizeof(float));
    dotProductSegmented<<<nBlocks, nThreads>>>(d_inputs, model->d_weights[0], d_product, batch_size, model->layer_size[0], model->layer_size[0], model->layer_size[1], model->d_biases[0]);
    cudaDeviceSynchronize();
    float *prod = new float[batch_size*model->layer_size[1]];
    for(int i = 0; i < n_inputs*model->layer_size[1]; i++) {
        prod[i] = 1.0f;
    }
    cudaMemcpy(prod, d_product, batch_size*model->layer_size[1]*sizeof(float), cudaMemcpyDeviceToHost);
    for(int i = 0; i < 8; i++) {
        EXPECT_FLOAT_EQ(prod[i], correctForward[i]);
    }
}

TEST_F(Example2Suite, TestForwardPass_Thread4_BatchSize2) {
    int nWorkers = 1;
    int nThreadsPerWorker = 1;
    int batch_size = 2;
    model->setupGPU(nThreadsPerWorker*nWorkers, batch_size);
    float *d_product;
    cudaMalloc(&d_inputs, n_inputs*model->layer_size[0]*sizeof(float));
    cudaMemcpy(d_inputs, input, n_inputs*model->layer_size[0]*sizeof(float), cudaMemcpyHostToDevice);
    cudaMalloc(&d_product, batch_size*model->layer_size[1]*sizeof(float));
    dim3 nBlocks(nWorkers, batch_size, 1);
    dim3 nThreads(nThreadsPerWorker, model->layer_size[0], 1);
    dotProductSegmented<<<nBlocks, nThreads>>>(d_inputs, model->d_weights[0], d_product, batch_size, model->layer_size[0], model->layer_size[0], model->layer_size[1], model->d_biases[0]);
    cudaDeviceSynchronize();
    float *prod = new float[batch_size*model->layer_size[1]];
    for(int i = 0; i < n_inputs*model->layer_size[1]; i++) {
        prod[i] = 1.0f;
    }
    cudaMemcpy(prod, d_product, batch_size*model->layer_size[1]*sizeof(float), cudaMemcpyDeviceToHost);
    for(int i = 0; i < 8; i++) {
        EXPECT_FLOAT_EQ(prod[i], correctForward[i]);
    }
}

TEST_F(Example2Suite, TestForwardPass_SingleThread_BatchSize1) {
    int nWorkers = 1;
    int nThreadsPerWorker = 1;
    int batch_size = 1;
    model->setupGPU(nThreadsPerWorker*nWorkers, batch_size);
    cudaMalloc(&d_inputs, n_inputs*model->layer_size[0]*sizeof(float));
    cudaMemcpy(d_inputs, input, n_inputs*model->layer_size[0]*sizeof(float), cudaMemcpyHostToDevice);
    std::shared_ptr<float> activations = model->forward_pass(d_inputs, n_inputs, batch_size, nWorkers, nThreadsPerWorker);
    for(int j = 0; j < 18; j++) {
        EXPECT_FLOAT_EQ(activations.get()[j], expected[j]);
    }
}

TEST_F(Example2Suite, TestForwardPass_SingleThread_BatchSize2) {
    int nWorkers = 1;
    int nThreadsPerWorker = 1;
    int batch_size = 2;
    model->setupGPU(nThreadsPerWorker*nWorkers, batch_size);
    cudaMalloc(&d_inputs, n_inputs*model->layer_size[0]*sizeof(float));
    cudaMemcpy(d_inputs, input, n_inputs*model->layer_size[0]*sizeof(float), cudaMemcpyHostToDevice);
    std::shared_ptr<float> activations = model->forward_pass(d_inputs, n_inputs, batch_size, nWorkers, nThreadsPerWorker);
    for(int j = 0; j < 18; j++) {
        EXPECT_FLOAT_EQ(activations.get()[j], expected[j]);
    }
}

TEST_F(Example2Suite, TestForwardPass_MultiThread_BatchSize1) {
    int nWorkers = 1;
    int nThreadsPerWorker = 1;
    int batch_size = 2;
    model->setupGPU(nThreadsPerWorker*nWorkers, batch_size);
    //need to set the threads for each block
    forward_pass_gridDim = new std::vector<dim3>(model->nLayers);
    forward_pass_blockDim = new std::vector<dim3>(model->nLayers);
    for(int i = 0; i < model->nLayers; i++) {
        forward_pass_gridDim[i].y = batch_size;
        forward_pass_blockDim[i].y = model->layer_size[i+1];
    }
    model->forward_pass_gridDim = forward_pass_gridDim;
    model->forward_pass_blockDim = forward_pass_blockDim;
    cudaMalloc(&d_inputs, n_inputs*model->layer_size[0]*sizeof(float));
    cudaMemcpy(d_inputs, input, n_inputs*model->layer_size[0]*sizeof(float), cudaMemcpyHostToDevice);
    std::shared_ptr<float> activations = model->forward_pass(d_inputs, n_inputs, batch_size, nWorkers, nThreadsPerWorker);
    for(int j = 0; j < 18; j++) {
        EXPECT_FLOAT_EQ(activations.get()[j], expected[j]);
    }
}
