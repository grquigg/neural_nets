#include <gtest/gtest.h>
#include "../include/lin_alg.h"
#include "../include/utils.h"
#include "../include/models.h"

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

TEST(SegmentedDotProduct, DotProductMultiThreadedEx1) {
  int nWorkers = 1, nThreadsPerWorker = 1, batch_size = 2;
  int layers[2] = {1, 2};
  float correct[4] = {0.413f, 0.326f, 0.442f, 0.384f};
  float **weights = new float*[1]{new float[2]{0.1f, 0.2f}};
  float **biases = new float*[1]{new float[2]{0.4f, 0.3f}};
  NeuralNetwork* model = new NeuralNetwork(1, layers, weights, biases, 1.0f);
  NeuralNetwork* d_model = copyModelToGPU(model, nWorkers, nThreadsPerWorker);
  float *input = new float[2]{0.13f, 0.42f};
  float *d_input;
  float *d_product;
  cudaMalloc(&d_input, batch_size*sizeof(float));
  cudaMemcpy(d_input, input, batch_size*sizeof(float), cudaMemcpyHostToDevice);
  cudaMalloc(&d_product, 4*sizeof(float));
  dim3 nBlocks(nWorkers, 1, 1);
  dim3 nThreads(nThreadsPerWorker, 2, 1);
  dotProductSegmented<<<nBlocks, nThreads>>>(d_input, d_model->weights[0], d_product, batch_size, model->layer_size[0], model->layer_size[0], model->layer_size[1], d_model->biases[0]);
  cudaDeviceSynchronize();
  float *prod = new float[4];
  cudaMemcpy(prod, d_product, 4*sizeof(float), cudaMemcpyDeviceToHost);
  for(int i = 0; i < 4; i++) {
    EXPECT_FLOAT_EQ(prod[i], correct[i]);
  }
}

TEST(SegmentedDotProduct, DotProductMultiThreadedEx2) {
  int nWorkers = 1;
  int nThreadsPerWorker = 1;
  int batch_size = 2;
  int layers[2] = {2, 4};
  float correct[8] = {0.74f, 1.1192f, 0.3564f, 0.8744f, 0.55250f, 0.81380f, 0.17610f, 0.60410f};
  float **weights = new float*[1]{new float[8]{0.15f, 0.1f, 0.19f, 0.35f, 0.4f, 0.54f, 0.42f, 0.68f}};
  float **biases = new float*[1]{new float[4]{0.42f, 0.72f, 0.01f, 0.3f}};
  NeuralNetwork model(1, layers, weights, biases, 1.0f);
  float input[4] = {0.32f, 0.68f, 0.83f, 0.02f};
  float *d_weights;
  float *d_input;
  float *d_product;
  float *d_bias;
  cudaMalloc(&d_bias, model.layer_size[1]*sizeof(float));
  cudaMemcpy(d_bias, model.biases[0], model.layer_size[1]*sizeof(float), cudaMemcpyHostToDevice);
  cudaMalloc(&d_weights, model.layer_size[0]*model.layer_size[1]*sizeof(float));
  cudaMemcpy(d_weights, model.weights[0], model.layer_size[0]*model.layer_size[1]*sizeof(float), cudaMemcpyHostToDevice);
  cudaMalloc(&d_input, batch_size*model.layer_size[0]*sizeof(float));
  cudaMemcpy(d_input, input, batch_size*model.layer_size[0]*sizeof(float), cudaMemcpyHostToDevice);
  cudaMalloc(&d_product, batch_size*model.layer_size[0]*model.layer_size[1]*sizeof(float));
  dim3 nBlocks(nWorkers, 1, 1);
  dim3 nThreads(nThreadsPerWorker, 2, 1);
  dotProductSegmented<<<nBlocks, nThreads>>>(d_input, d_weights, d_product, batch_size, model.layer_size[0], model.layer_size[0], model.layer_size[1], d_bias);
  cudaDeviceSynchronize();
  float *prod = new float[batch_size*model.layer_size[0]*model.layer_size[1]];
  cudaMemcpy(prod, d_product, 8*sizeof(float), cudaMemcpyDeviceToHost);
  for(int i = 0; i < 8; i++) {
    EXPECT_FLOAT_EQ(prod[i], correct[i]);
  }

  nBlocks.y = 1;
  nThreads.y = 2;
  for(int i = 0; i < 8; i++) {
    prod[i] = 0;
  }
  cudaMemcpy(d_product, prod, 8*sizeof(float), cudaMemcpyHostToDevice);
  dotProductSegmented<<<nBlocks, nThreads>>>(d_input, d_weights, d_product, 2, model.layer_size[0], model.layer_size[0], model.layer_size[1], d_bias);
  cudaDeviceSynchronize();

  cudaMemcpy(prod, d_product, 8*sizeof(float), cudaMemcpyDeviceToHost);
  for(int i = 0; i < 8; i++) {
    EXPECT_FLOAT_EQ(prod[i], correct[i]);
  }

  nBlocks.y = 2;
  for(int i = 0; i < 8; i++) {
    prod[i] = 0;
  }
  cudaMemcpy(d_product, prod, 8*sizeof(float), cudaMemcpyHostToDevice);
  dotProductSegmented<<<nBlocks, nThreads>>>(d_input, d_weights, d_product, 2, model.layer_size[0], model.layer_size[0], model.layer_size[1], d_bias);
  cudaDeviceSynchronize();
  cudaMemcpy(prod, d_product, 8*sizeof(float), cudaMemcpyDeviceToHost);
  for(int i = 0; i < 8; i++) {
    EXPECT_FLOAT_EQ(prod[i], correct[i]);
  }

  nThreads.z = 2;
  for(int i = 0; i < 8; i++) {
    prod[i] = 0;
  }
  cudaMemcpy(d_product, prod, 8*sizeof(float), cudaMemcpyHostToDevice);
  dotProductSegmented<<<nBlocks, nThreads>>>(d_input, d_weights, d_product, 2, model.layer_size[0], model.layer_size[0], model.layer_size[1], d_bias);
  cudaDeviceSynchronize();
  cudaMemcpy(prod, d_product, 8*sizeof(float), cudaMemcpyDeviceToHost);
  for(int i = 0; i < 8; i++) {
    EXPECT_FLOAT_EQ(prod[i], correct[i]);
  }
}

TEST(SegmentedSigmoid, MultiThreadedEx1) {
  int nWorkers = 1;
  int nThreadsPerWorker = 2;
  int batch_size = 2;
  int layers[2] = {1, 2};
  float correct[4] = {0.601807f, 0.58078581f, 0.6087355f, 0.59483749f};
  float **weights = new float*[1]{new float[2]{0.1f, 0.2f}};
  float **biases = new float*[1]{new float[2]{0.4f, 0.3f}};
  NeuralNetwork model(1, layers, weights, biases, 1.0f);
  float input[2] = {0.13f, 0.42f};
  float *d_weights;
  float *d_input;
  float *d_product;
  float *d_bias;
  cudaMalloc(&d_bias, model.layer_size[1]*sizeof(float));
  cudaMemcpy(d_bias, model.biases[0], model.layer_size[1]*sizeof(float), cudaMemcpyHostToDevice);
  cudaMalloc(&d_weights, model.layer_size[1]*sizeof(float));
  cudaMemcpy(d_weights, model.weights[0], model.layer_size[0]*model.layer_size[1]*sizeof(float), cudaMemcpyHostToDevice);
  cudaMalloc(&d_input, batch_size*sizeof(float));
  cudaMemcpy(d_input, input, batch_size*sizeof(float), cudaMemcpyHostToDevice);
  cudaMalloc(&d_product, batch_size*model.layer_size[1]*sizeof(float));
  dim3 nBlocks(nWorkers, 1, 1);
  dim3 nThreads(nThreadsPerWorker, 1, 1);
  dotProductSegmented<<<nBlocks, nThreads>>>(d_input, d_weights, d_product, batch_size, model.layer_size[0], model.layer_size[0], model.layer_size[1], d_bias);
  cudaDeviceSynchronize();
  sigmoidSegmented<<<nWorkers, nThreadsPerWorker>>>(d_product, batch_size*model.layer_size[1]);
  cudaDeviceSynchronize();
  float *prod = new float[batch_size*model.layer_size[1]];
  cudaMemcpy(prod, d_product, batch_size*model.layer_size[1]*sizeof(float), cudaMemcpyDeviceToHost);
  for(int i = 0; i < batch_size*model.layer_size[1]; i++) {
    EXPECT_FLOAT_EQ(prod[i], correct[i]);
  }
}

TEST(SegmentedSigmoid, MultiThreadedEx2) {
  int nWorkers = 2;
  int nThreadsPerWorker = 1;
  int batch_size = 2;
  int layers[2] = {2, 4};
  float correct[8] = {0.67699581f, 0.75384f, 0.58816868f, 0.7056604f, 0.63471538f, 0.69291866f, 0.54391158f, 0.64659375f};
  float **weights = new float*[1]{new float[8]{0.15f, 0.1f, 0.19f, 0.35f, 0.4f, 0.54f, 0.42f, 0.68f}};
  float **biases = new float*[1]{new float[4]{0.42f, 0.72f, 0.01f, 0.3f}};
  NeuralNetwork model(1, layers, weights, biases, 1.0f);
  float input[4] = {0.32f, 0.68f, 0.83f, 0.02f};
  float *d_weights;
  float *d_input;
  float *d_product;
  float *d_bias;
  cudaMalloc(&d_bias, model.layer_size[1]*sizeof(float));
  cudaMemcpy(d_bias, model.biases[0], model.layer_size[1]*sizeof(float), cudaMemcpyHostToDevice);
  cudaMalloc(&d_weights, model.layer_size[0]*model.layer_size[1]*sizeof(float));
  cudaMemcpy(d_weights, model.weights[0], model.layer_size[0]*model.layer_size[1]*sizeof(float), cudaMemcpyHostToDevice);
  cudaMalloc(&d_input, batch_size*model.layer_size[0]*sizeof(float));
  cudaMemcpy(d_input, input, batch_size*model.layer_size[0]*sizeof(float), cudaMemcpyHostToDevice);
  cudaMalloc(&d_product, batch_size*model.layer_size[1]*sizeof(float));
  dim3 nBlocks(nWorkers, 1, 1);
  dim3 nThreads(nThreadsPerWorker, 1, 1);
  dotProductSegmented<<<nBlocks, nThreads>>>(d_input, d_weights, d_product, batch_size, model.layer_size[0], model.layer_size[0], model.layer_size[1], d_bias);
  cudaDeviceSynchronize();
  sigmoidSegmented<<<nWorkers, nThreadsPerWorker>>>(d_product, batch_size*model.layer_size[1]);
  cudaDeviceSynchronize();
  float *prod = new float[batch_size*model.layer_size[1]];
  cudaMemcpy(prod, d_product, batch_size*model.layer_size[1]*sizeof(float), cudaMemcpyDeviceToHost);
  for(int i = 0; i < batch_size*model.layer_size[1]; i++) {
    printf("Value of i: %d\n", i);
    EXPECT_FLOAT_EQ(prod[i], correct[i]);
  }
}

// TEST(ForwardPass, MultiThreadedDotProduct2Ex1_BATCH_SIZE_1) {
//   int nWorkers = 1;
//   int nThreadsPerWorker = 1;
//   int batch_size = 1;
//   float correctOutput[6] = {0.601807f, 0.58078581f, 1.349375f, 0.6087355f, 0.59483749f, 1.3612702f};
//   float input[2] = {0.13000f, 0.42f};
//   int layers[3] = {1, 2, 1};
//   float *weights[2];
//   float weight0[2] = {0.1f, 0.2f};
//   float weight1[2] = {0.5f, 0.6f};
//   weights[0] = weight0;
//   weights[1] = weight1;
//   float *biases[2];
//   float bias0[2] = {0.4f, 0.3f};
//   float bias1[1] = {0.7f};
//   biases[0] = bias0;
//   biases[1] = bias1;
//   NeuralNetwork* model = new NeuralNetwork(2, layers, weights, biases, 1.0f);
//   float *d_weights0;
//   float *d_weights1;
//   float *d_input;
//   float *d_bias0;
//   float *d_bias1;
//   cudaMalloc(&d_bias0, model->layer_size[1]*sizeof(float));
//   cudaMemcpy(d_bias0, model->biases[0], model->layer_size[1]*sizeof(float), cudaMemcpyHostToDevice);
//   cudaMalloc(&d_bias1, model->layer_size[2]*sizeof(float));
//   cudaMemcpy(d_bias1, model->biases[1], model->layer_size[2]*sizeof(float), cudaMemcpyHostToDevice);
//   cudaMalloc(&d_weights0, model->layer_size[0]*model->layer_size[1]*sizeof(float));
//   cudaMemcpy(d_weights0, model->weights[0], model->layer_size[0]*model->layer_size[1]*sizeof(float), cudaMemcpyHostToDevice);
//   cudaMalloc(&d_weights1, model->layer_size[1]*model->layer_size[2]*sizeof(float));
//   cudaMemcpy(d_weights1, model->weights[1], model->layer_size[1]*model->layer_size[2]*sizeof(float), cudaMemcpyHostToDevice);
//   cudaMalloc(&d_input, 2*sizeof(float));
//   cudaMemcpy(d_input, input, 2*sizeof(float), cudaMemcpyHostToDevice);
//   int activations_size = 0;
//   int * offsets = new int[model->nLayers];
//   for(int i = 1; i <= model->nLayers; i++) {
//     offsets[i-1] = (batch_size * activations_size);
//     // printf("Offset at %d: %d\n", i-1, offsets[i-1]);
//     activations_size += model->layer_size[i];
//   }
//   EXPECT_EQ(offsets[0], 0);
//   EXPECT_EQ(offsets[1], 2);
//   float * d_activations = new float[batch_size*activations_size];
//   float * activations = new float[batch_size*activations_size];
//   //device pointers
//   int * d_offsets;
//   cudaMalloc(&d_activations, activations_size*batch_size*sizeof(float));
//   cudaMalloc(&d_offsets, model->nLayers*sizeof(int));
//   for(int i = 0; i < activations_size*batch_size; i++) {
//     activations[i] = 1;
//   }
//   cudaMemcpy(d_activations, activations, activations_size*batch_size*sizeof(float), cudaMemcpyHostToDevice);
//   cudaMemcpy(d_offsets, offsets, model->nLayers*sizeof(int), cudaMemcpyHostToDevice);
//   dim3 nBlocks(nWorkers, 1, 1);
//   dim3 nThreads(nThreadsPerWorker, 1, 1);
//   for(int i = 0; i < 2; i+=1) {
//     printf("Bad result\n");
//     dotProductSegmented<<<nBlocks, nThreads>>>(d_input+(i*model->layer_size[0]), d_weights0, d_activations, batch_size, model->layer_size[0], model->layer_size[0], model->layer_size[1], d_bias0);
//     cudaDeviceSynchronize();
//     printf("NEXT\n");
//     sigmoidSegmented<<<nWorkers, nThreadsPerWorker>>>(d_activations, batch_size*model->layer_size[1]);
//     cudaDeviceSynchronize();
//     printf("NEXT\n");
//     dotProductSegmented<<<nBlocks, nThreads>>>(d_activations, d_weights1, d_activations+(offsets[1]*batch_size), batch_size, model->layer_size[1], model->layer_size[1], model->layer_size[2], d_bias1);
//     cudaDeviceSynchronize();
//     printf("NEXT\n");
//     cudaMemcpy(activations, d_activations, activations_size*batch_size*sizeof(float), cudaMemcpyDeviceToHost);
//     for(int j = 0; j < activations_size; j++) {
//         printf("j: %d\n", j);
//         EXPECT_FLOAT_EQ(correctOutput[i*activations_size+j], activations[j]);
//     }
//   }
// }

TEST(ForwardPass, MultiThreadedThreadedDotProduct2Ex1_BATCH_SIZE_2) {
  int nWorkers = 2;
  int nThreadsPerWorker = 1;
  int batch_size = 2;
  int nLayers = 2;
  float correctOutput[6] = {0.601807f, 0.58078581f, 0.6087355f, 0.59483749f, 1.349375f, 1.3612702f};
  float input[2] = {0.13000f, 0.42f};
  int *layers = new int[nLayers+1]{1, 2, 1};
  float **weights = new float*[2];
  weights[0] = new float[2]{0.1f, 0.2f};
  weights[1] = new float[2]{0.5f, 0.6f};
  float **biases = new float*[2];
  biases[0] = new float[2]{0.4f, 0.3f};
  biases[1] = new float[1]{0.7f};
  NeuralNetwork model(nLayers, layers, weights, biases, 1.0f);
  float *d_input;
  NeuralNetwork* d_model = copyModelToGPU(&model, nWorkers, nThreadsPerWorker);
  cudaMalloc(&d_input, 2*model.layer_size[0]*sizeof(float));
  cudaMemcpy(d_input, input, 2*model.layer_size[0]*sizeof(float), cudaMemcpyHostToDevice);
  int activations_size = 0;
  int * offsets = new int[model.nLayers];
  for(int i = 1; i <= model.nLayers; i++) {
    offsets[i-1] = (batch_size * activations_size);
    // printf("Offset at %d: %d\n", i-1, offsets[i-1]);
    activations_size += model.layer_size[i];
  }
  EXPECT_EQ(offsets[0], 0);
  EXPECT_EQ(offsets[1], 4);
  EXPECT_EQ(activations_size*batch_size, 6);
  float * d_activations = new float[batch_size*activations_size];
  float * activations = new float[batch_size*activations_size];
  //device pointers
  int * d_offsets;
  cudaMalloc(&d_activations, activations_size*batch_size*sizeof(float));
  cudaMalloc(&d_offsets, model.nLayers*sizeof(int));
  for(int i = 0; i < activations_size*batch_size; i++) {
    activations[i] = 1;
  }
  cudaMemcpy(d_activations, activations, activations_size*batch_size*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_offsets, offsets, model.nLayers*sizeof(int), cudaMemcpyHostToDevice);
  dim3 nBlocks(nWorkers, 1, 1);
  dim3 nThreads(nThreadsPerWorker, 1, 1);
  for(int i = 0; i < 2; i+=batch_size) {
    dotProductSegmented<<<nBlocks, nThreads>>>(d_input+(i*model.layer_size[0]), d_model->weights[0], d_activations, batch_size, model.layer_size[0], model.layer_size[0], model.layer_size[1], d_model->biases[0]);
    cudaDeviceSynchronize();
    sigmoidSegmented<<<nWorkers, nThreadsPerWorker>>>(d_activations, batch_size*model.layer_size[1]);
    cudaDeviceSynchronize();
    dotProductSegmented<<<nBlocks, nThreads>>>(d_activations, d_model->weights[1], d_activations+(offsets[1]), batch_size, model.layer_size[1], model.layer_size[1], model.layer_size[2], d_model->biases[1]);
    cudaDeviceSynchronize();
    cudaMemcpy(activations, d_activations, activations_size*batch_size*sizeof(float), cudaMemcpyDeviceToHost);
    for(int j = 0; j < activations_size*batch_size; j++) {
        EXPECT_FLOAT_EQ(correctOutput[i*activations_size+j], activations[j]);
    }
  }
}

TEST(ForwardPass, MultiThreadedSoftmaxEx1_BATCH_SIZE_1) {
  float correctOutput[2] = {1.0f, 1.0f};
  int nWorkers = 1;
  int nThreadsPerWorker = 1;
  int batch_size = 1;
  int nLayers = 2;
  float input[2] = {0.13000f, 0.42f};
  int *layers = new int[nLayers+1]{1, 2, 1};
  float **weights = new float*[2];
  weights[0] = new float[2]{0.1f, 0.2f};
  weights[1] = new float[2]{0.5f, 0.6f};
  float **biases = new float*[2];
  biases[0] = new float[2]{0.4f, 0.3f};
  biases[1] = new float[1]{0.7f};
  NeuralNetwork *model = new NeuralNetwork(nLayers, layers, weights, biases, 1.0f);
  float *d_input;
  NeuralNetwork* d_model = copyModelToGPU(model, nWorkers, nThreadsPerWorker);
  cudaMalloc(&d_input, 2*model->layer_size[0]*sizeof(float));
  cudaMemcpy(d_input, input, 2*model->layer_size[0]*sizeof(float), cudaMemcpyHostToDevice);
  int activations_size = 0;
  int * offsets = new int[model->nLayers];
  for(int i = 1; i <= model->nLayers; i++) {
    offsets[i-1] = (batch_size * activations_size);
    // printf("Offset at %d: %d\n", i-1, offsets[i-1]);
    activations_size += model->layer_size[i];
  }
  EXPECT_EQ(offsets[0], 0);
  EXPECT_EQ(offsets[1], 2);
  EXPECT_EQ(activations_size*batch_size, 3);
  float * d_activations = new float[batch_size*activations_size];
  float * activations = new float[batch_size*activations_size];
  //device pointers
  int * d_offsets;
  cudaMalloc(&d_activations, activations_size*batch_size*sizeof(float));
  cudaMalloc(&d_offsets, model->nLayers*sizeof(int));
  for(int i = 0; i < activations_size*batch_size; i++) {
    activations[i] = 1;
  }
  cudaMemcpy(d_activations, activations, activations_size*batch_size*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_offsets, offsets, model->nLayers*sizeof(int), cudaMemcpyHostToDevice);
  dim3 nBlocks(nWorkers, 1, 1);
  dim3 nThreads(nThreadsPerWorker, 2, 1);
  for(int i = 0; i < 2; i+=batch_size) {
    nBlocks.y = 1;
    nThreads.y = 2;
    dotProductSegmented<<<nBlocks, nThreads>>>(d_input+(i*model->layer_size[0]), d_model->weights[0], d_activations, batch_size, model->layer_size[0], model->layer_size[0], model->layer_size[1], d_model->biases[0]);
    cudaDeviceSynchronize();
    sigmoidSegmented<<<nWorkers, nThreadsPerWorker>>>(d_activations, batch_size*model->layer_size[1]);
    cudaDeviceSynchronize();
    nThreads.y = 1;
    dotProductSegmented<<<nBlocks, nThreads>>>(d_activations, d_model->weights[1], d_activations+(offsets[1]), batch_size, model->layer_size[1], model->layer_size[1], model->layer_size[2], d_model->biases[1]);
    cudaDeviceSynchronize();
    softmaxSegmented<<<nWorkers, nThreadsPerWorker>>>(d_activations+(offsets[1]*batch_size), batch_size, model->layer_size[2]);
    cudaDeviceSynchronize();
    cudaMemcpy(activations, d_activations, activations_size*batch_size*sizeof(float), cudaMemcpyDeviceToHost);
    EXPECT_FLOAT_EQ(correctOutput[i], activations[2]);
  }
}

TEST(ForwardPass, MultiThreadedSoftmaxEx1_BATCH_SIZE_2) {
  float correctOutput[2] = {1.0f, 1.0f};
  int nWorkers = 2;
  int nThreadsPerWorker = 1;
  int batch_size = 1;
  int nLayers = 2;
  float input[2] = {0.13000f, 0.42f};
  int *layers = new int[nLayers+1]{1, 2, 1};
  float **weights = new float*[2];
  weights[0] = new float[2]{0.1f, 0.2f};
  weights[1] = new float[2]{0.5f, 0.6f};
  float **biases = new float*[2];
  biases[0] = new float[2]{0.4f, 0.3f};
  biases[1] = new float[1]{0.7f};
  NeuralNetwork *model = new NeuralNetwork(nLayers, layers, weights, biases, 1.0f);
  float *d_input;
  NeuralNetwork* d_model = copyModelToGPU(model, nWorkers, nThreadsPerWorker);
  cudaMalloc(&d_input, 2*model->layer_size[0]*sizeof(float));
  cudaMemcpy(d_input, input, 2*model->layer_size[0]*sizeof(float), cudaMemcpyHostToDevice);
  int activations_size = 0;
  int * offsets = new int[model->nLayers];
  for(int i = 1; i <= model->nLayers; i++) {
    offsets[i-1] = (batch_size * activations_size);
    // printf("Offset at %d: %d\n", i-1, offsets[i-1]);
    activations_size += model->layer_size[i];
  }
  EXPECT_EQ(offsets[0], 0);
  EXPECT_EQ(offsets[1], 2);
  EXPECT_EQ(activations_size*batch_size, 3);
  float * d_activations = new float[batch_size*activations_size];
  float * activations = new float[batch_size*activations_size];
  //device pointers
  int * d_offsets;
  cudaMalloc(&d_activations, activations_size*batch_size*sizeof(float));
  cudaMalloc(&d_offsets, model->nLayers*sizeof(int));
  for(int i = 0; i < activations_size*batch_size; i++) {
    activations[i] = 1;
  }
  cudaMemcpy(d_activations, activations, activations_size*batch_size*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_offsets, offsets, model->nLayers*sizeof(int), cudaMemcpyHostToDevice);
  dim3 nBlocks(nWorkers, 1, 1);
  dim3 nThreads(nThreadsPerWorker, 2, 1);
  for(int i = 0; i < 2; i+=batch_size) {
    dotProductSegmented<<<nBlocks, nThreads>>>(d_input+(i*model->layer_size[0]), d_model->weights[0], d_activations, batch_size, model->layer_size[0], model->layer_size[0], model->layer_size[1], d_model->biases[0]);
    sigmoidSegmented<<<nWorkers, nThreadsPerWorker>>>(d_activations, batch_size*model->layer_size[1]);
    cudaDeviceSynchronize();
    nThreads.y = 1;
    dotProductSegmented<<<nBlocks, nThreads>>>(d_activations, d_model->weights[1], d_activations+(offsets[1]), batch_size, model->layer_size[1], model->layer_size[1], model->layer_size[2], d_model->biases[1]);
    cudaDeviceSynchronize();
    softmaxSegmented<<<nWorkers, nThreadsPerWorker>>>(d_activations+(offsets[1]), batch_size, model->layer_size[2]);
    cudaDeviceSynchronize();
    cudaMemcpy(activations, d_activations, activations_size*batch_size*sizeof(float), cudaMemcpyDeviceToHost);
    for(int j = 0; j < batch_size; j++) {
        EXPECT_FLOAT_EQ(correctOutput[j], activations[offsets[1]+j]);
    }
  }
}

TEST(ForwardPass, MultiThreadedForwardPassEx2_BATCH_SIZE_1) {
  int nWorkers = 1;
  int nThreadsPerWorker = 1;
  int batch_size = 1;
  int nLayers = 3;
  int layers[4] = {2, 4, 3, 2};
  float ** weights = new float*[3];
  weights[0] = new float[8]{0.15f, 0.1f, 0.19f, 0.35f, 0.4f, 0.54f, 0.42f, 0.68f};
  weights[1] = new float[12]{0.67f, 0.42f, 0.56f, 0.14f, 0.2f, 0.8f, 0.96f, 0.32f, 0.69f, 0.87f, 0.89f, 0.09f};
  weights[2] = new float[6]{0.87f, 0.1f, 0.42f, 0.95f, 0.53f, 0.69f};
  float ** biases = new float*[3];
  biases[0] = new float[4]{0.42f, 0.72f, 0.01f, 0.3f};
  biases[1] = new float[3]{0.21f, 0.87f, 0.03f};
  biases[2] = new float[2]{0.04f, 0.17f};
  float input[4] = {0.32f, 0.68f, 0.83f, 0.02f};
  float expected[18] = {
    0.67699581f, 0.75384f, 0.58816868f, 0.7056604f,
    0.87519467f, 0.8929618f, 0.81480443f,
    0.48506981f, 0.51493f,
    0.63471538f, 0.69291866f, 0.54391158f, 0.64659375f,
    0.86020094f, 0.8833645f, 0.79790765f,
    0.4841319f, 0.51586807f
  };
  float * d_input;
  NeuralNetwork* model = new NeuralNetwork(nLayers, layers, weights, biases, 1.0f);
  NeuralNetwork* d_model = copyModelToGPU(model, nWorkers, nThreadsPerWorker);
  cudaMalloc(&d_input, 2*model->layer_size[0]*sizeof(float));
  cudaMemcpy(d_input, input, 2*model->layer_size[0]*sizeof(float), cudaMemcpyHostToDevice);
  int activations_size = 0;
  int * offsets = new int[model->nLayers];
  for(int i = 1; i <= model->nLayers; i++) {
    offsets[i-1] = (batch_size * activations_size);
    // printf("Offset at %d: %d\n", i-1, offsets[i-1]);
    activations_size += model->layer_size[i];
    
  }
  EXPECT_EQ(offsets[0], 0);
  EXPECT_EQ(offsets[1], batch_size*model->layer_size[1]);
  EXPECT_EQ(offsets[2], offsets[1]+(batch_size*model->layer_size[2]));
  float * d_activations = new float[batch_size*activations_size];
  float * activations = new float[batch_size*activations_size];
  //device pointers
  int * d_offsets;
  cudaMalloc(&d_activations, activations_size*batch_size*sizeof(float));
  cudaMalloc(&d_offsets, model->nLayers*sizeof(int));
  for(int i = 0; i < activations_size*batch_size; i++) {
    activations[i] = 1;
  }
  cudaMemcpy(d_activations, activations, activations_size*batch_size*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_offsets, offsets, model->nLayers*sizeof(int), cudaMemcpyHostToDevice);
  dim3 nBlocks(nWorkers, 1, 1);
  dim3 nThreads(nThreadsPerWorker, 4, 1);
  for(int i = 0; i < 2; i+=batch_size) {
    dotProductSegmented<<<nBlocks, nThreads>>>(d_input+(i*model->layer_size[0]), d_model->weights[0], d_activations, batch_size, model->layer_size[0], model->layer_size[0], model->layer_size[1], d_model->biases[0]);
    cudaDeviceSynchronize();
    sigmoidSegmented<<<nWorkers, nThreadsPerWorker>>>(d_activations, batch_size*model->layer_size[1]);
    cudaDeviceSynchronize();
    nThreads.y = 3;
    dotProductSegmented<<<nBlocks, nThreads>>>(d_activations, d_model->weights[1], d_activations+(offsets[1]), batch_size, model->layer_size[1], model->layer_size[1], model->layer_size[2], d_model->biases[1]);
    cudaDeviceSynchronize();
    sigmoidSegmented<<<nWorkers, nThreadsPerWorker>>>(d_activations+offsets[1], batch_size*model->layer_size[2]);
    cudaDeviceSynchronize();
    nThreads.y = 2;
    dotProductSegmented<<<nBlocks, nThreads>>>(d_activations+offsets[1], d_model->weights[2], d_activations+offsets[2], batch_size, model->layer_size[2], model->layer_size[2], model->layer_size[3], d_model->biases[2]);
    cudaDeviceSynchronize();
    softmaxSegmented<<<nWorkers, nThreadsPerWorker>>>(d_activations+(offsets[2]), batch_size, model->layer_size[3]);
    cudaMemcpy(activations, d_activations, activations_size*batch_size*sizeof(float), cudaMemcpyDeviceToHost);
    printf("BATCH %d\n", i);
    for(int j = 0; j < activations_size*batch_size; j++) {
      printf("j: %d\n", i*activations_size+j);
      EXPECT_FLOAT_EQ(expected[i*activations_size+j], activations[j]);
    }
  }
}

TEST(ForwardPass, MultiThreadedForwardPassEx2_BATCH_SIZE_2) {
  int nWorkers = 1;
  int nThreadsPerWorker = 1;
  int batch_size = 2;
  int nLayers = 3;
  int layers[4] = {2, 4, 3, 2};
  float ** weights = new float*[3];
  weights[0] = new float[8]{0.15f, 0.1f, 0.19f, 0.35f, 0.4f, 0.54f, 0.42f, 0.68f};
  weights[1] = new float[12]{0.67f, 0.42f, 0.56f, 0.14f, 0.2f, 0.8f, 0.96f, 0.32f, 0.69f, 0.87f, 0.89f, 0.09f};
  weights[2] = new float[6]{0.87f, 0.1f, 0.42f, 0.95f, 0.53f, 0.69f};
  float ** biases = new float*[3];
  biases[0] = new float[4]{0.42f, 0.72f, 0.01f, 0.3f};
  biases[1] = new float[3]{0.21f, 0.87f, 0.03f};
  biases[2] = new float[2]{0.04f, 0.17f};
  float input[4] = {0.32f, 0.68f, 0.83f, 0.02f};
  float expected[18] = {
    0.67699581f, 0.75384f, 0.58816868f, 0.7056604f,
    0.63471538f, 0.69291866f, 0.54391158f, 0.64659375f,
    0.87519467f, 0.8929618f, 0.81480443f,
    0.86020094f, 0.8833645f, 0.79790765f,
    0.48506981f, 0.51493f,
    0.4841319f, 0.51586807f
  };
  float * d_input;
  NeuralNetwork* model = new NeuralNetwork(nLayers, layers, weights, biases, 1.0f);
  NeuralNetwork* d_model = copyModelToGPU(model, nWorkers, nThreadsPerWorker);
  cudaMalloc(&d_input, 2*model->layer_size[0]*sizeof(float));
  cudaMemcpy(d_input, input, 2*model->layer_size[0]*sizeof(float), cudaMemcpyHostToDevice);
  int activations_size = 0;
  int * offsets = new int[model->nLayers];
  for(int i = 1; i <= model->nLayers; i++) {
    offsets[i-1] = (batch_size * activations_size);
    // printf("Offset at %d: %d\n", i-1, offsets[i-1]);
    activations_size += model->layer_size[i];
    
  }
  EXPECT_EQ(offsets[0], 0);
  EXPECT_EQ(offsets[1], batch_size*model->layer_size[1]);
  EXPECT_EQ(offsets[2], offsets[1]+(batch_size*model->layer_size[2]));
  float * d_activations = new float[batch_size*activations_size];
  float * activations = new float[batch_size*activations_size];
  //device pointers
  int * d_offsets;
  cudaMalloc(&d_activations, activations_size*batch_size*sizeof(float));
  cudaMalloc(&d_offsets, model->nLayers*sizeof(int));
  for(int i = 0; i < activations_size*batch_size; i++) {
    activations[i] = 1;
  }
  cudaMemcpy(d_activations, activations, activations_size*batch_size*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_offsets, offsets, model->nLayers*sizeof(int), cudaMemcpyHostToDevice);
  dim3 nBlocks(nWorkers, 1, 1);
  dim3 nThreads(nThreadsPerWorker, 4, 1);
  for(int i = 0; i < 2; i+=batch_size) {
    dotProductSegmented<<<nBlocks, nThreads>>>(d_input+(i*model->layer_size[0]), d_model->weights[0], d_activations, batch_size, model->layer_size[0], model->layer_size[0], model->layer_size[1], d_model->biases[0]);
    cudaDeviceSynchronize();
    sigmoidSegmented<<<nWorkers, nThreadsPerWorker>>>(d_activations, batch_size*model->layer_size[1]);
    cudaDeviceSynchronize();
    nThreads.y = 3;
    dotProductSegmented<<<nBlocks, nThreads>>>(d_activations, d_model->weights[1], d_activations+(offsets[1]), batch_size, model->layer_size[1], model->layer_size[1], model->layer_size[2], d_model->biases[1]);
    cudaDeviceSynchronize();
    sigmoidSegmented<<<nWorkers, nThreadsPerWorker>>>(d_activations+offsets[1], batch_size*model->layer_size[2]);
    cudaDeviceSynchronize();
    nThreads.y = 2;
    dotProductSegmented<<<nBlocks, nThreads>>>(d_activations+offsets[1], d_model->weights[2], d_activations+offsets[2], batch_size, model->layer_size[2], model->layer_size[2], model->layer_size[3], d_model->biases[2]);
    cudaDeviceSynchronize();
    softmaxSegmented<<<nWorkers, nThreadsPerWorker>>>(d_activations+(offsets[2]), batch_size, model->layer_size[3]);
    cudaMemcpy(activations, d_activations, activations_size*batch_size*sizeof(float), cudaMemcpyDeviceToHost);
    printf("BATCH %d\n", i);
    for(int j = 0; j < activations_size*batch_size; j++) {
      printf("j: %d\n", i*activations_size+j);
      EXPECT_FLOAT_EQ(expected[i*activations_size+j], activations[j]);
    }
  }
}

TEST(ForwardPass, MultiThreadedForwardPassEx2_BATCH_SIZE_2_1) {
  int nWorkers = 1;
  int nThreadsPerWorker = 1;
  int batch_size = 2;
  int nLayers = 3;
  int layers[4] = {2, 4, 3, 2};
  float ** weights = new float*[3];
  weights[0] = new float[8]{0.15f, 0.1f, 0.19f, 0.35f, 0.4f, 0.54f, 0.42f, 0.68f};
  weights[1] = new float[12]{0.67f, 0.42f, 0.56f, 0.14f, 0.2f, 0.8f, 0.96f, 0.32f, 0.69f, 0.87f, 0.89f, 0.09f};
  weights[2] = new float[6]{0.87f, 0.1f, 0.42f, 0.95f, 0.53f, 0.69f};
  float ** biases = new float*[3];
  biases[0] = new float[4]{0.42f, 0.72f, 0.01f, 0.3f};
  biases[1] = new float[3]{0.21f, 0.87f, 0.03f};
  biases[2] = new float[2]{0.04f, 0.17f};
  float input[4] = {0.32f, 0.68f, 0.83f, 0.02f};
  float expected[18] = {
    0.67699581f, 0.75384f, 0.58816868f, 0.7056604f,
    0.63471538f, 0.69291866f, 0.54391158f, 0.64659375f,
    0.87519467f, 0.8929618f, 0.81480443f,
    0.86020094f, 0.8833645f, 0.79790765f,
    0.48506981f, 0.51493f,
    0.4841319f, 0.51586807f
  };
  float * d_input;
  NeuralNetwork* model = new NeuralNetwork(nLayers, layers, weights, biases, 1.0f);
  NeuralNetwork* d_model = copyModelToGPU(model, nWorkers, nThreadsPerWorker);
  cudaMalloc(&d_input, 2*model->layer_size[0]*sizeof(float));
  cudaMemcpy(d_input, input, 2*model->layer_size[0]*sizeof(float), cudaMemcpyHostToDevice);
  int activations_size = 0;
  int * offsets = new int[model->nLayers];
  for(int i = 1; i <= model->nLayers; i++) {
    offsets[i-1] = (batch_size * activations_size);
    // printf("Offset at %d: %d\n", i-1, offsets[i-1]);
    activations_size += model->layer_size[i];
    
  }
  EXPECT_EQ(offsets[0], 0);
  EXPECT_EQ(offsets[1], batch_size*model->layer_size[1]);
  EXPECT_EQ(offsets[2], offsets[1]+(batch_size*model->layer_size[2]));
  float * d_activations = new float[batch_size*activations_size];
  float * activations = new float[batch_size*activations_size];
  //device pointers
  int * d_offsets;
  cudaMalloc(&d_activations, activations_size*batch_size*sizeof(float));
  cudaMalloc(&d_offsets, model->nLayers*sizeof(int));
  for(int i = 0; i < activations_size*batch_size; i++) {
    activations[i] = 1;
  }
  cudaMemcpy(d_activations, activations, activations_size*batch_size*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_offsets, offsets, model->nLayers*sizeof(int), cudaMemcpyHostToDevice);
  dim3 nBlocks(nWorkers, 1, 1);
  dim3 nThreads(nThreadsPerWorker, 2, 1);
  for(int i = 0; i < 2; i+=batch_size) {
    dotProductSegmented<<<nBlocks, nThreads>>>(d_input+(i*model->layer_size[0]), d_model->weights[0], d_activations, batch_size, model->layer_size[0], model->layer_size[0], model->layer_size[1], d_model->biases[0]);
    cudaDeviceSynchronize();
    sigmoidSegmented<<<nWorkers, nThreadsPerWorker>>>(d_activations, batch_size*model->layer_size[1]);
    cudaDeviceSynchronize();
    nThreads.y = 3;
    dotProductSegmented<<<nBlocks, nThreads>>>(d_activations, d_model->weights[1], d_activations+(offsets[1]), batch_size, model->layer_size[1], model->layer_size[1], model->layer_size[2], d_model->biases[1]);
    cudaDeviceSynchronize();
    sigmoidSegmented<<<nWorkers, nThreadsPerWorker>>>(d_activations+offsets[1], batch_size*model->layer_size[2]);
    cudaDeviceSynchronize();
    nThreads.y = 2;
    dotProductSegmented<<<nBlocks, nThreads>>>(d_activations+offsets[1], d_model->weights[2], d_activations+offsets[2], batch_size, model->layer_size[2], model->layer_size[2], model->layer_size[3], d_model->biases[2]);
    cudaDeviceSynchronize();
    softmaxSegmented<<<nWorkers, nThreadsPerWorker>>>(d_activations+(offsets[2]), batch_size, model->layer_size[3]);
    cudaMemcpy(activations, d_activations, activations_size*batch_size*sizeof(float), cudaMemcpyDeviceToHost);
    printf("BATCH %d\n", i);
    for(int j = 0; j < activations_size*batch_size; j++) {
      printf("j: %d\n", i*activations_size+j);
      EXPECT_FLOAT_EQ(expected[i*activations_size+j], activations[j]);
    }
  }
}