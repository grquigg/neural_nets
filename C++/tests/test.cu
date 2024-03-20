#include <gtest/gtest.h>
#include "../include/lin_alg.h"
#include "../include/utils.h"
#include "../include/models.h"
// Demonstrate some basic assertions.

NeuralNetwork* buildModel(int nLayers, int * layer_size, float** weights, float **biases, float lambda, int nThreadsPerWorker, int nWorkers) {
    NeuralNetwork* model = new NeuralNetwork;
    model->nLayers = nLayers;
    model->lambda = lambda;
    model->layer_size = (int*)malloc((model->nLayers+1)*sizeof(int));
    model->layer_size = layer_size;
    model->weights = (float**)malloc((model->nLayers)*sizeof(float*));
    model->biases = (float**)malloc((model->nLayers)*sizeof(float*));
    model->gradients = (float**)malloc((model->nLayers)*sizeof(float*));
    model->grad_biases =(float**)malloc((model->nLayers)*sizeof(float*));
    model->nClasses = model->layer_size[model->nLayers];
    for(int i = 1; i < model->nLayers+1; i++) {
        model->weights[i-1] = initializeFlatRandomArray(model->layer_size[i-1], model->layer_size[i]);
        model->biases[i-1] = initializeFlatRandomArray(1, model->layer_size[i]);
        model->grad_biases[i-1] = (float*)malloc(nThreadsPerWorker*nWorkers*model->layer_size[i]);
        model->gradients[i-1] =(float*)malloc(nThreadsPerWorker*nWorkers*model->layer_size[i-1]*model->layer_size[i]);
    }
    model->weights = weights;
    model->biases = biases;
    return model;
}

TEST(Main, TestBuildModel) { 
  int nWorkers = 1;
  int nThreadsPerWorker = 1;
  int layers[2] = {1, 2};
  float *weights[1];
  weights[0] = (float*)malloc(2*sizeof(float));
  weights[0][0] = 0.1f;
  weights[0][1] = 0.2f;
  float *biases[1];
  biases[0] = (float*)malloc(2*sizeof(float));
  biases[0][0] = 0.4f;
  biases[0][1] = 0.3f;
  NeuralNetwork* model = buildModel(1, layers, weights, biases, 1.0, 1, 1);
  EXPECT_EQ(model->weights[0][0], 0.1f);
  EXPECT_EQ(model->weights[0][1], 0.2f);
  EXPECT_EQ(model->biases[0][0], 0.4f);
  EXPECT_EQ(model->biases[0][1], 0.3f);
  EXPECT_EQ(model->nLayers, 1);
  EXPECT_EQ(model->layer_size[0], 1);
  EXPECT_EQ(model->layer_size[1], 2);
  // NeuralNetwork* d_model = copyModelToGPU(model, 1, 1);
}

TEST(SegmentedDotProduct, SingleThreaded) {
    int nWorkers = 1, nThreadsPerWorker = 1;
    float arr1[6] = {1,2,3,4,5,6};
    float arr2[12] = {-1,-2,-3,-4,-5,-6,-7,-8,-9,-10,-11,-12};
    float product[8];
    float correct_ans[8] = {-38.0f, -44.0f, -50.0f, -56.0f, -83.0f, -98.0f, -113.0f, -128.0f};
    float *darr1;
    float *darr2;
    float *dproduct;
    cudaMalloc(&darr1, 6*sizeof(float));
    cudaMalloc(&darr2, 12*sizeof(float));
    cudaMalloc(&dproduct, 8*sizeof(float));
    cudaMemcpy(darr1, arr1, 6*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(darr2, arr2, 12*sizeof(float), cudaMemcpyHostToDevice);
    dim3 nBlocks(nWorkers, 1, 1);
    dim3 nThreads(nThreadsPerWorker, 1, 1);
    dotProductSegmented<<<nBlocks, nThreads>>>(darr1, darr2, dproduct, 2, 3, 3, 4);
    cudaDeviceSynchronize();
    cudaMemcpy(product, dproduct, 8*sizeof(float), cudaMemcpyDeviceToHost);
    for(int i = 0; i < 2; i++) {
        for(int j = 0; j < 4; j++) {
            EXPECT_EQ(product[i*4+j], correct_ans[i*4+j]);
        }
    }
}

TEST(SegmentedDotProduct, MultiThreaded) {
    int nWorkers = 1, nThreadsPerWorker = 1;
    float arr1[6] = {1,2,3,4,5,6};
    float arr2[12] = {-1,-2,-3,-4,-5,-6,-7,-8,-9,-10,-11,-12};
    float product[8];
    float correct_ans[8] = {-38.0f, -44.0f, -50.0f, -56.0f, -83.0f, -98.0f, -113.0f, -128.0f};
    float *darr1;
    float *darr2;
    float *dproduct;
    cudaMalloc(&darr1, 6*sizeof(float));
    cudaMalloc(&darr2, 12*sizeof(float));
    cudaMalloc(&dproduct, 8*sizeof(float));
    cudaMemcpy(darr1, arr1, 6*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(darr2, arr2, 12*sizeof(float), cudaMemcpyHostToDevice);
    dim3 nBlocks(nWorkers, 2, 1);
    dim3 nThreads(nThreadsPerWorker, 2, 1);
    dotProductSegmented<<<nBlocks, nThreads>>>(darr1, darr2, dproduct, 2, 3, 3, 4);
    cudaDeviceSynchronize();
    cudaMemcpy(product, dproduct, 8*sizeof(float), cudaMemcpyDeviceToHost);
    for(int i = 0; i < 2; i++) {
        for(int j = 0; j < 4; j++) {
            EXPECT_EQ(product[i*4+j], correct_ans[i*4+j]);
        }
    }
}

TEST(SegmentedDotProduct, DotProductSingleThreadedEx1) { //this is based on the Backprop example 1 from 589 HW4
  int nWorkers = 1;
  int nThreadsPerWorker = 1;
  int layers[2] = {1, 2};
  float correct[4] = {0.413f, 0.326f, 0.442f, 0.384f};
  float *weights[1];
  weights[0] = (float*)malloc(2*sizeof(float));
  weights[0][0] = 0.1f;
  weights[0][1] = 0.2f;
  float *biases[1];
  biases[0] = (float*)malloc(2*sizeof(float));
  biases[0][0] = 0.4f;
  biases[0][1] = 0.3f;
  NeuralNetwork* model = buildModel(1, layers, weights, biases, 1.0, 1, 1);
  float input[2] = {0.13f, 0.42f};
  float product[4] = {1.0f, 1.0f, 1.0f, 1.0f};
  float *d_weights;
  float *d_input;
  float *d_product;
  float *d_bias;
  cudaMalloc(&d_bias, 2*sizeof(float));
  cudaMemcpy(d_bias, model->bias[0], 2*sizeof(float), cudaMemcpyHostToDevice);
  cudaMalloc(&d_weights, 2*sizeof(float));
  cudaMemcpy(d_weights, model->weights[0], 2*sizeof(float), cudaMemcpyHostToDevice);
  cudaMalloc(&d_input, 2*sizeof(float));
  cudaMemcpy(d_input, input, 2*sizeof(float), cudaMemcpyHostToDevice);
  cudaMalloc(&d_product, 4*sizeof(float));
  dim3 nBlocks(nWorkers, 1, 1);
  dim3 nThreads(nThreadsPerWorker, 1, 1);
  dotProductSegmented<<<nBlocks, nThreads>>>(d_input, d_weights, d_product, 2, model->layer_size[0], model->layer_size[0], model->layer_size[1], d_bias);
}