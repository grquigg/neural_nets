#include <gtest/gtest.h>
#include "models.h"
#include "lin_alg.h"
#include "example2.h"

TEST_F(Example2Suite, TestForwardPass1_Thread1_BatchSize1_Ex1) {
    int nWorkers = 1;
    int nThreadsPerWorker = 1;
    dim3 nBlocks(nWorkers, 1, 1);
    dim3 nThreads(nThreadsPerWorker, 1, 1);
    int batch_size = 1;
    model->setupGPU(nThreadsPerWorker*nWorkers, batch_size);
    float *d_product;
    d_inputs = transferMatrixToDevice(input, batch_size, model->layer_size[0]);
    cudaMalloc(&d_product, batch_size*model->layer_size[1]*sizeof(float));
    dotProductSegmented<<<nBlocks, nThreads>>>(d_inputs.get(), model->d_weights[0], d_product, batch_size, model->layer_size[0], model->layer_size[0], model->layer_size[1], model->d_biases[0]);
    cudaDeviceSynchronize();
    float *prod = new float[batch_size*model->layer_size[0]*model->layer_size[1]];
    for(int i = 0; i < n_inputs*model->layer_size[1]; i++) {
        prod[i] = 1.0f;
    }
    cudaMemcpy(prod, d_product, batch_size*model->layer_size[1]*sizeof(float), cudaMemcpyDeviceToHost);
    for(int i = 0; i < 4; i++) {
        EXPECT_FLOAT_EQ(prod[i], correctForward[0][i]);
    }
    //check to make sure that the dot product only modifies the parts of the array that we care about
    for(int i = 4; i < 8; i++) {
        EXPECT_FLOAT_EQ(prod[i], 1.0f);
    }
}

TEST_F(Example2Suite, TestForwardPass1_Thread1_BatchSize1_Ex2) {
    int nWorkers = 1;
    int nThreadsPerWorker = 1;
    dim3 nBlocks(nWorkers, 1, 1);
    dim3 nThreads(nThreadsPerWorker, 1, 1);
    int batch_size = 1;
    model->setupGPU(nThreadsPerWorker*nWorkers, batch_size);
    cudaDeviceSynchronize();
    float *d_product;
    std::cout << *(input+model->layer_size[0]) << std::endl;
    d_inputs = transferMatrixToDevice(input+model->layer_size[0], batch_size, model->layer_size[0]);
    cudaMalloc(&d_product, batch_size*model->layer_size[1]*sizeof(float));
    dotProductSegmented<<<nBlocks, nThreads>>>(d_inputs.get(), model->d_weights[0], d_product, batch_size, model->layer_size[0], model->layer_size[0], model->layer_size[1], model->d_biases[0]);
    std::cout << "Successful" << std::endl;
    cudaDeviceSynchronize();
    float *prod = new float[batch_size*model->layer_size[1]];
    for(int i = 0; i < batch_size*model->layer_size[1]; i++) {
        prod[i] = 1.0f;
    }
    cudaMemcpy(prod, d_product, batch_size*model->layer_size[1]*sizeof(float), cudaMemcpyDeviceToHost);
    for(int i = 0; i < 4; i++) {
        EXPECT_FLOAT_EQ(prod[i], correctForward[0][i+model->layer_size[1]]);
    }
}

TEST_F(Example2Suite, TestForwardPass1_Thread1_BatchSize2) {
    int nWorkers = 1;
    int nThreadsPerWorker = 1;
    dim3 nBlocks(nWorkers, 1, 1);
    dim3 nThreads(nThreadsPerWorker, 1, 1);
    int batch_size = 2;
    model->setupGPU(nThreadsPerWorker*nWorkers, batch_size);
    float *d_product;
    d_inputs = transferMatrixToDevice(input, batch_size, model->layer_size[0]);
    cudaMalloc(&d_product, batch_size*model->layer_size[1]*sizeof(float));
    dotProductSegmented<<<nBlocks, nThreads>>>(d_inputs.get(), model->d_weights[0], d_product, batch_size, model->layer_size[0], model->layer_size[0], model->layer_size[1], model->d_biases[0]);
    cudaDeviceSynchronize();
    float *prod = new float[batch_size*model->layer_size[1]];
    for(int i = 0; i < n_inputs*model->layer_size[1]; i++) {
        prod[i] = 1.0f;
    }
    cudaMemcpy(prod, d_product, batch_size*model->layer_size[1]*sizeof(float), cudaMemcpyDeviceToHost);
    for(int i = 0; i < 8; i++) {
        EXPECT_FLOAT_EQ(prod[i], correctForward[0][i]);
    }
}

TEST_F(Example2Suite, TestForwardPass1_Thread2_BatchSize2) {
    int nWorkers = 2;
    int nThreadsPerWorker = 1;
    dim3 nBlocks(nWorkers, 1, 1);
    dim3 nThreads(nThreadsPerWorker, 1, 1);
    int batch_size = 2;
    model->setupGPU(nThreadsPerWorker*nWorkers, batch_size);
    float *d_product;
    d_inputs = transferMatrixToDevice(input, batch_size, model->layer_size[0]);
    cudaMalloc(&d_product, batch_size*model->layer_size[1]*sizeof(float));
    dotProductSegmented<<<nBlocks, nThreads>>>(d_inputs.get(), model->d_weights[0], d_product, batch_size, model->layer_size[0], model->layer_size[0], model->layer_size[1], model->d_biases[0]);
    cudaDeviceSynchronize();
    float *prod = new float[batch_size*model->layer_size[1]];
    for(int i = 0; i < n_inputs*model->layer_size[1]; i++) {
        prod[i] = 1.0f;
    }
    cudaMemcpy(prod, d_product, batch_size*model->layer_size[1]*sizeof(float), cudaMemcpyDeviceToHost);
    for(int i = 0; i < 8; i++) {
        EXPECT_FLOAT_EQ(prod[i], correctForward[0][i]);
    }
}

TEST_F(Example2Suite, TestForwardPass1_Thread4_BatchSize2) {
    int nWorkers = 1;
    int nThreadsPerWorker = 1;
    int batch_size = 2;
    model->setupGPU(nThreadsPerWorker*nWorkers, batch_size);
    float *d_product;
    d_inputs = transferMatrixToDevice(input, batch_size, model->layer_size[0]);
    cudaMalloc(&d_product, batch_size*model->layer_size[1]*sizeof(float));
    dim3 nBlocks(nWorkers, batch_size, 1);
    dim3 nThreads(nThreadsPerWorker, model->layer_size[0], 1);
    dotProductSegmented<<<nBlocks, nThreads>>>(d_inputs.get(), model->d_weights[0], d_product, batch_size, model->layer_size[0], model->layer_size[0], model->layer_size[1], model->d_biases[0]);
    cudaDeviceSynchronize();
    float *prod = new float[batch_size*model->layer_size[1]];
    for(int i = 0; i < n_inputs*model->layer_size[1]; i++) {
        prod[i] = 1.0f;
    }
    cudaMemcpy(prod, d_product, batch_size*model->layer_size[1]*sizeof(float), cudaMemcpyDeviceToHost);
    for(int i = 0; i < 8; i++) {
        EXPECT_FLOAT_EQ(prod[i], correctForward[0][i]);
    }
}

TEST_F(Example2Suite, TestForwardPass1_Thread2_BatchSize1_Ex1) {
    int nWorkers = 1;
    int nThreadsPerWorker = 1;
    int batch_size = 1;
    model->setupGPU(nThreadsPerWorker*nWorkers, batch_size);
    float *d_product;
    d_inputs = transferMatrixToDevice(input, batch_size, model->layer_size[0]);
    cudaMalloc(&d_product, batch_size*model->layer_size[1]*sizeof(float));
    dim3 nBlocks(nWorkers, batch_size, 1);
    dim3 nThreads(nThreadsPerWorker, model->layer_size[0], 1);
    dotProductSegmented<<<nBlocks, nThreads>>>(d_inputs.get(), model->d_weights[0], d_product, batch_size, model->layer_size[0], model->layer_size[0], model->layer_size[1], model->d_biases[0]);
    cudaDeviceSynchronize();
    float *prod = new float[batch_size*model->layer_size[1]];
    for(int i = 0; i < batch_size*model->layer_size[1]; i++) {
        prod[i] = 1.0f;
    }
    cudaMemcpy(prod, d_product, batch_size*model->layer_size[1]*sizeof(float), cudaMemcpyDeviceToHost);
    for(int i = 0; i < 4; i++) {
        EXPECT_FLOAT_EQ(prod[i], correctForward[0][i]);
    }
}

TEST_F(Example2Suite, TestForwardPass1_Thread2_BatchSize1_Ex2) {
    int nWorkers = 1;
    int nThreadsPerWorker = 1;
    int batch_size = 1;
    model->setupGPU(nThreadsPerWorker*nWorkers, batch_size);
    cudaDeviceSynchronize();
    float *d_product;
    std::cout << *(input+model->layer_size[0]) << std::endl;
    d_inputs = transferMatrixToDevice(input+model->layer_size[0], batch_size, model->layer_size[0]);
    dim3 nBlocks(nWorkers, batch_size, 1);
    dim3 nThreads(nThreadsPerWorker, model->layer_size[0], 1);
    cudaMalloc(&d_product, batch_size*model->layer_size[1]*sizeof(float));
    dotProductSegmented<<<nBlocks, nThreads>>>(d_inputs.get(), model->d_weights[0], d_product, batch_size, model->layer_size[0], model->layer_size[0], model->layer_size[1], model->d_biases[0]);
    std::cout << "Successful" << std::endl;
    cudaDeviceSynchronize();
    float *prod = new float[batch_size*model->layer_size[1]];
    for(int i = 0; i < batch_size*model->layer_size[1]; i++) {
        prod[i] = 1.0f;
    }
    cudaMemcpy(prod, d_product, batch_size*model->layer_size[1]*sizeof(float), cudaMemcpyDeviceToHost);
    for(int i = 0; i < 4; i++) {
        EXPECT_FLOAT_EQ(prod[i], correctForward[0][i+model->layer_size[1]]);
    }
}
