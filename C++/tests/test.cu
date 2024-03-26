#include <gtest/gtest.h>
#include "../include/lin_alg.h"
#include "../include/utils.h"
#include "../include/models.h"
// Demonstrate some basic assertions.

NeuralNetwork * copyModel(NeuralNetwork *model, int nWorkers, int nThreadsPerWorker) {
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
    float **temp_biases = new float*[model->nLayers];
    float **temp_gradients = new float*[model->nLayers];
    float **temp_grad_biases = new float*[model->nLayers];
    for(int i = 1; i < model->nLayers+1; i++) {
        cudaMalloc(&temp_weights[i-1], model->layer_size[i-1]*model->layer_size[i]*sizeof(float));
        cudaMemcpy(temp_weights[i-1], model->weights[i-1], model->layer_size[i-1]*model->layer_size[i]*sizeof(float), cudaMemcpyHostToDevice);
        cudaMalloc(&temp_biases[i-1], model->layer_size[i]*sizeof(float));
        cudaMemcpy(temp_biases[i-1], model->biases[i-1], model->layer_size[i]*sizeof(float), cudaMemcpyHostToDevice);
        cudaMalloc(&temp_gradients[i-1], nThreadsPerWorker*nWorkers*model->layer_size[i-1]*model->layer_size[i]*sizeof(float));
        // cudaMemcpy(temp_gradients[i-1], model->gradients[i-1], nThreadsPerWorker*nWorkers*model->layer_size[i-1]*model->layer_size[i]*sizeof(float), cudaMemcpyHostToDevice);
        cudaMalloc(&temp_grad_biases[i-1],  nThreadsPerWorker*nWorkers*model->layer_size[i]*sizeof(float));
    }
    printf("Success\n");
    cudaMalloc(&d_gradients, (model->nLayers)*sizeof(float*));
    cudaMemcpy(d_gradients, temp_gradients, (model->nLayers)*sizeof(float*), cudaMemcpyHostToDevice);
    cudaMalloc(&d_grad_biases, (model->nLayers)*sizeof(float*));
    cudaMemcpy(d_grad_biases, temp_grad_biases, (model->nLayers)*sizeof(float*), cudaMemcpyHostToDevice);
    cudaMalloc(&d_biases, (model->nLayers)*sizeof(float*));
    cudaMemcpy(d_biases, temp_biases, (model->nLayers)*sizeof(float*), cudaMemcpyHostToDevice);
    cudaMalloc(&d_weights, (model->nLayers)*sizeof(float*));
    cudaMemcpy(d_weights, temp_weights, (model->nLayers)*sizeof(float*), cudaMemcpyHostToDevice);
    NeuralNetwork temp = *model;
    temp.nClasses = model->nClasses;
    temp.nLayers = model->nLayers;
    temp.layer_size = nLayers;
    temp.weights = d_weights;
    temp.gradients = d_gradients;
    temp.biases = d_biases;
    temp.grad_biases = d_grad_biases;
    temp.lambda = model->lambda;
    cudaMemcpy(d_model, &temp, sizeof(NeuralNetwork), cudaMemcpyHostToDevice);
    return d_model;
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
}

TEST(Main, TestCopyModelToGPU) {
  int nWorkers = 1;
  int nThreadsPerWorker = 1;
  int nLayers = 1;
  float regularization = 1.0f;
  int layers[2] = {1, 2};
  float *weights[1];
  float weight[2] = {0.1f, 0.2f};
  weights[0] = weight;
  float *biases[1];
  float bias[2] = {0.4f, 0.3f};
  biases[0] = bias;
  NeuralNetwork* model = buildModel(nLayers, layers, weights, biases, regularization, nWorkers, nThreadsPerWorker);
  NeuralNetwork* d_model = copyModelToGPU(model, nWorkers, nThreadsPerWorker);
  NeuralNetwork *copiedModel = new NeuralNetwork;
  NeuralNetwork *temp = new NeuralNetwork;
  temp->weights = (float**)malloc(sizeof(float*));
  temp->biases = (float**)malloc(sizeof(float*));
  temp->layer_size = (int*)malloc(2*sizeof(int));
  float * temp_weights = (float*)malloc(2*sizeof(float));
  float * temp_biases = (float*)malloc(2*sizeof(float));
  cudaMemcpy(copiedModel, d_model, sizeof(NeuralNetwork), cudaMemcpyDeviceToHost);
  cudaMemcpy(temp->weights, copiedModel->weights, sizeof(float**), cudaMemcpyDeviceToHost);
  cudaMemcpy(temp->biases, copiedModel->biases, sizeof(float**), cudaMemcpyDeviceToHost);
  cudaMemcpy(temp_weights, temp->weights[0], 2*sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(temp_biases, temp->biases[0], 2*sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(temp->layer_size, copiedModel->layer_size, 2*sizeof(int), cudaMemcpyDeviceToHost);
  EXPECT_FLOAT_EQ(temp_weights[0], 0.1f);
  EXPECT_FLOAT_EQ(temp_weights[1], 0.2f);
  EXPECT_EQ(temp->layer_size[0], 1);
  EXPECT_EQ(temp->layer_size[1], 2);
  EXPECT_FLOAT_EQ(temp_biases[0], 0.4f);
  EXPECT_FLOAT_EQ(temp_biases[1], 0.3f);
  printf("Success\n");

}
