#include <gtest/gtest.h>
#include "../include/lin_alg.h"
#include "../include/utils.h"
#include "../include/models.h"
#include <memory>

TEST(SegmentedDotProduct, DotProductSingleThreadedEx1) { //this is based on the Backprop example 1 from 589 HW4
  int nWorkers = 1, nThreadsPerWorker = 1, batch_size = 2;
  dim3 nBlocks(nWorkers, 1, 1);
  dim3 nThreads(nThreadsPerWorker, 1, 1);
  int * layers = new int[2]{1,2};
  float correct[4] = {0.413f, 0.326f, 0.442f, 0.384f};
  float **weights = new float*[1]{new float[2]{0.1f, 0.2f}};
  float **biases = new float*[1]{new float[2]{0.4f, 0.3f}};
  NeuralNetwork model(1, layers, weights, biases, 1.0f);
  model.setupGPU(1, batch_size);
  float *input = new float[2]{0.13f, 0.42f};

  std::shared_ptr<float> d_input = transferMatrixToDevice(input, batch_size, 1);
  float *d_product;
  cudaMalloc(&d_product, 4*sizeof(float));
  dotProductSegmented<<<nBlocks, nThreads>>>(d_input.get(), model.d_weights[0], d_product, batch_size, model.layer_size[0], model.layer_size[0], model.layer_size[1], model.d_biases[0]);
  cudaDeviceSynchronize();
  float *prod = new float[4];
  cudaMemcpy(prod, d_product, 4*sizeof(float), cudaMemcpyDeviceToHost);
  for(int i = 0; i < 4; i++) {
    EXPECT_FLOAT_EQ(prod[i], correct[i]);
  }
  cudaFree(d_product);
  free(weights[0]);
  free(biases[0]);
  free(weights);
  free(biases);
  free(input);
  free(layers);
  free(prod);
}

TEST(SegmentedDotProduct, DotProductSingleThreadedEx2) { //REMEMBER THAT WE'RE TAKING THE TRANSPOSE OF THE MATRIX IN THE EXAMPLE
  int nWorkers = 1;
  int nThreadsPerWorker = 1;
  dim3 nBlocks(nWorkers, 1, 1);
  dim3 nThreads(nThreadsPerWorker, 1, 1);
  int batch_size = 2;
  int layers[2] = {2, 4};
  float correct[8] = {0.74f, 1.1192f, 0.3564f, 0.8744f, 0.55250f, 0.81380f, 0.17610f, 0.60410f};
  float **weights = new float*[1]{new float[8]{0.15f, 0.1f, 0.19f, 0.35f, 0.4f, 0.54f, 0.42f, 0.68f}};
  float **biases = new float*[1]{new float[4]{0.42f, 0.72f, 0.01f, 0.3f}};
  NeuralNetwork model(1, layers, weights, biases, 1.0f);
  model.setupGPU(nThreadsPerWorker*nWorkers, batch_size);
  float input[4] = {0.32f, 0.68f, 0.83f, 0.02f};
  std::shared_ptr<float> d_weights = transferMatrixToDevice(model.d_weights[0], model.layer_size[0], model.layer_size[1]);
  std::shared_ptr<float> d_inputs = transferMatrixToDevice(input, 2, 2);
  std::shared_ptr<float> d_bias = transferMatrixToDevice(model.d_biases[0], model.layer_size[1], 1);
  float *d_product;
  cudaMalloc(&d_product, batch_size*model.layer_size[1]*sizeof(float));
  dotProductSegmented<<<nBlocks, nThreads>>>(d_inputs.get(), d_weights.get(), d_product, batch_size, model.layer_size[0], model.layer_size[0], model.layer_size[1], d_bias.get());
  cudaDeviceSynchronize();
  float *prod = new float[batch_size*model.layer_size[0]*model.layer_size[1]];
  cudaMemcpy(prod, d_product, 8*sizeof(float), cudaMemcpyDeviceToHost);
  for(int i = 0; i < 8; i++) {
    EXPECT_FLOAT_EQ(prod[i], correct[i]);
  }
  free(prod);
  cudaFree(d_product);
  free(weights[0]);
  free(biases[0]);
  free(weights);
  free(biases);
}

TEST(SegmentedSigmoid, SingleThreadedEx1) {
  int nWorkers = 1;
  int nThreadsPerWorker = 1;
  dim3 nBlocks(nWorkers, 1, 1);
  dim3 nThreads(nThreadsPerWorker, 1, 1);
  int batch_size = 2;
  int layers[2] = {1, 2};
  float correct[4] = {0.601807f, 0.58078581f, 0.6087355f, 0.59483749f};
  float **weights = new float*[1]{new float[2]{0.1f, 0.2f}};
  float **biases = new float*[1]{new float[2]{0.4f, 0.3f}};
  NeuralNetwork model(1, layers, weights, biases, 1.0f);
  model.setupGPU(nThreadsPerWorker*nWorkers, batch_size);
  float input[2] = {0.13f, 0.42f};
  std::shared_ptr<float> d_weights = transferMatrixToDevice(weights[0], model.layer_size[0], model.layer_size[1]);
  std::shared_ptr<float> d_input = transferMatrixToDevice(input, 2, 1);
  std::shared_ptr<float> d_bias = transferMatrixToDevice(biases[0], 1, 2);
  float *d_product;
  cudaMalloc(&d_product, batch_size*model.layer_size[1]*sizeof(float));
  dotProductSegmented<<<nBlocks, nThreads>>>(d_input.get(), d_weights.get(), d_product, batch_size, model.layer_size[0], model.layer_size[0], model.layer_size[1], d_bias.get());
  cudaDeviceSynchronize();
  sigmoidSegmented<<<nWorkers, nThreadsPerWorker>>>(d_product, batch_size*model.layer_size[1]);
  cudaDeviceSynchronize();
  float *prod = new float[batch_size*model.layer_size[1]];
  cudaMemcpy(prod, d_product, batch_size*model.layer_size[1]*sizeof(float), cudaMemcpyDeviceToHost);
  for(int i = 0; i < batch_size*model.layer_size[1]; i++) {
    EXPECT_FLOAT_EQ(prod[i], correct[i]);
  }
  free(prod);
  cudaFree(d_product);
  free(weights[0]);
  free(biases[0]);
  free(weights);
  free(biases);
}

TEST(SegmentedSigmoid, SingleThreadedEx2) {
  int nWorkers = 1;
  int nThreadsPerWorker = 1;
  dim3 nBlocks(nWorkers, 1, 1);
  dim3 nThreads(nThreadsPerWorker, 1, 1);
  int batch_size = 2;
  int layers[2] = {2, 4};
  float correct[8] = {0.67699581f, 0.75384f, 0.58816868f, 0.7056604f, 0.63471538f, 0.69291866f, 0.54391158f, 0.64659375f};
  float **weights = new float*[1]{new float[8]{0.15f, 0.1f, 0.19f, 0.35f, 0.4f, 0.54f, 0.42f, 0.68f}};
  float **biases = new float*[1]{new float[4]{0.42f, 0.72f, 0.01f, 0.3f}};
  NeuralNetwork model(1, layers, weights, biases, 1.0f);
  model.setupGPU(nThreadsPerWorker*nWorkers, batch_size);
  float input[4] = {0.32f, 0.68f, 0.83f, 0.02f};
  float *d_product;
  std::shared_ptr<float> d_weights = transferMatrixToDevice(weights[0], model.layer_size[0], model.layer_size[1]);
  std::shared_ptr<float> d_input = transferMatrixToDevice(input, 2, 2);
  std::shared_ptr<float> d_bias = transferMatrixToDevice(biases[0], 1, 4);
  cudaMalloc(&d_product, batch_size*model.layer_size[1]*sizeof(float));
  dotProductSegmented<<<nBlocks, nThreads>>>(d_input.get(), d_weights.get(), d_product, batch_size, model.layer_size[0], model.layer_size[0], model.layer_size[1], d_bias.get());
  cudaDeviceSynchronize();
  sigmoidSegmented<<<nWorkers, nThreadsPerWorker>>>(d_product, batch_size*model.layer_size[1]);
  cudaDeviceSynchronize();
  float *prod = new float[batch_size*model.layer_size[1]];
  cudaMemcpy(prod, d_product, batch_size*model.layer_size[1]*sizeof(float), cudaMemcpyDeviceToHost);
  for(int i = 0; i < batch_size*model.layer_size[1]; i++) {
    printf("Value of i: %d\n", i);
    EXPECT_FLOAT_EQ(prod[i], correct[i]);
  }
  free(prod);
  cudaFree(d_product);
  free(weights[0]);
  free(biases[0]);
  free(weights);
  free(biases);
}

TEST(ForwardPass, SingleThreadedDotProduct2Ex1_BATCH_SIZE_1) {
  int nWorkers = 1;
  int nThreadsPerWorker = 1;
  int batch_size = 1;
  int nLayers = 2;
  float correctOutput[6] = {0.601807f, 0.58078581f, 1.349375f, 0.6087355f, 0.59483749f, 1.3612702f};
  float input[2] = {0.13000f, 0.42f};
  int *layers = new int[nLayers+1]{1, 2, 1};
  float **weights = new float*[2];
  weights[0] = new float[2]{0.1f, 0.2f};
  weights[1] = new float[2]{0.5f, 0.6f};
  float **biases = new float*[2];
  biases[0] = new float[2]{0.4f, 0.3f};
  biases[1] = new float[1]{0.7f};
  NeuralNetwork model(nLayers, layers, weights, biases, 1.0f);
  model.setupGPU(nWorkers*nThreadsPerWorker, batch_size);
  std::shared_ptr<float> d_input = transferMatrixToDevice(input, 1, 2);
  int activations_size = 0;
  int * offsets = new int[model.nLayers];
  for(int i = 1; i <= model.nLayers; i++) {
    offsets[i-1] = (batch_size * activations_size);
    // printf("Offset at %d: %d\n", i-1, offsets[i-1]);
    activations_size += model.layer_size[i];
  }
  EXPECT_EQ(offsets[0], 0);
  EXPECT_EQ(offsets[1], 2);
  std::shared_ptr<int> d_offsets = transferMatrixToDevice(offsets, model.nLayers, 1);
  float * activations = new float[batch_size*activations_size];
  //device pointers
  for(int i = 0; i < activations_size*batch_size; i++) {
    activations[i] = 1;
  }
  std::shared_ptr<float> d_activations = transferMatrixToDevice(activations, batch_size, activations_size);
  dim3 nBlocks(nWorkers, 1, 1);
  dim3 nThreads(nThreadsPerWorker, 1, 1);
  std::cout << "Alright" << std::endl;
  for(int i = 0; i < 2; i+=1) {
    dotProductSegmented<<<nBlocks, nThreads>>>(d_input.get()+(i*model.layer_size[0]), model.d_weights[0], d_activations.get(), batch_size, model.layer_size[0], model.layer_size[0], model.layer_size[1], model.d_biases[0]);
    cudaDeviceSynchronize();
    sigmoidSegmented<<<nWorkers, nThreadsPerWorker>>>(d_activations.get(), batch_size*model.layer_size[1]);
    cudaDeviceSynchronize();
    dotProductSegmented<<<nBlocks, nThreads>>>(d_activations.get(), model.d_weights[1], d_activations.get()+(offsets[1]*batch_size), batch_size, model.layer_size[1], model.layer_size[1], model.layer_size[2], model.d_biases[1]);
    cudaDeviceSynchronize();
    cudaMemcpy(activations, d_activations.get(), activations_size*batch_size*sizeof(float), cudaMemcpyDeviceToHost);
    for(int j = 0; j < activations_size; j++) {
        EXPECT_FLOAT_EQ(correctOutput[i*activations_size+j], activations[j]);
    }
  }
  free(offsets);
  free(weights[0]);
  free(biases[0]);
  free(weights);
  free(biases);
  free(activations);
  free(layers);
}

TEST(ForwardPass, SingleThreadedDotProduct2Ex1_BATCH_SIZE_2) {
  int nWorkers = 1;
  int nThreadsPerWorker = 1;
  dim3 nBlocks(nWorkers, 1, 1);
  dim3 nThreads(nThreadsPerWorker, 1, 1);
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
  model.setupGPU(nWorkers*nThreadsPerWorker, batch_size);
  std::shared_ptr<float> d_input = transferMatrixToDevice(input, 1, 2);
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
  float * activations = new float[batch_size*activations_size];
  //device pointers
  std::shared_ptr<int> d_offsets = transferMatrixToDevice(offsets, model.nLayers, 1);
  for(int i = 0; i < activations_size*batch_size; i++) {
    activations[i] = 1;
  }
  std::shared_ptr<float> d_activations = transferMatrixToDevice(activations, batch_size, activations_size);
  for(int i = 0; i < 2; i+=batch_size) {
    dotProductSegmented<<<nBlocks, nThreads>>>(d_input.get()+(i*model.layer_size[0]), model.d_weights[0], d_activations.get(), batch_size, model.layer_size[0], model.layer_size[0], model.layer_size[1], model.d_biases[0]);
    cudaDeviceSynchronize();
    sigmoidSegmented<<<nWorkers, nThreadsPerWorker>>>(d_activations.get(), batch_size*model.layer_size[1]);
    cudaDeviceSynchronize();
    dotProductSegmented<<<nBlocks, nThreads>>>(d_activations.get(), model.d_weights[1], d_activations.get()+(offsets[1]), batch_size, model.layer_size[1], model.layer_size[1], model.layer_size[2], model.d_biases[1]);
    cudaDeviceSynchronize();
    cudaMemcpy(activations, d_activations.get(), activations_size*batch_size*sizeof(float), cudaMemcpyDeviceToHost);
    for(int j = 0; j < activations_size*batch_size; j++) {
        printf("j: %d\n", j);
        EXPECT_FLOAT_EQ(correctOutput[i*activations_size+j], activations[j]);
    }
  }
  free(offsets);
  free(weights[0]);
  free(biases[0]);
  free(weights);
  free(biases);
  free(activations);
  free(layers);
}

TEST(ForwardPass, SingleThreadedSoftmaxEx1_BATCH_SIZE_1) {
  float correctOutput[2] = {1.0f, 1.0f};
  int nWorkers = 1;
  int nThreadsPerWorker = 1;
  dim3 nBlocks(nWorkers, 1, 1);
  dim3 nThreads(nThreadsPerWorker, 1, 1);
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
  NeuralNetwork model(nLayers, layers, weights, biases, 1.0f);
  model.setupGPU(nWorkers*nThreadsPerWorker, batch_size);
  std::shared_ptr<float> d_input = transferMatrixToDevice(input, 2, model.layer_size[0]);
  int activations_size = 0;
  int * offsets = new int[model.nLayers];
  for(int i = 1; i <= model.nLayers; i++) {
    offsets[i-1] = (batch_size * activations_size);
    // printf("Offset at %d: %d\n", i-1, offsets[i-1]);
    activations_size += model.layer_size[i];
  }
  EXPECT_EQ(offsets[0], 0);
  EXPECT_EQ(offsets[1], 2);
  EXPECT_EQ(activations_size*batch_size, 3);
  float * activations = new float[batch_size*activations_size];
  //device pointers
  std::shared_ptr<int> d_offsets = transferMatrixToDevice(offsets, model.nLayers, 1);
  for(int i = 0; i < activations_size*batch_size; i++) {
    activations[i] = 1;
  }
  std::shared_ptr<float> d_activations = transferMatrixToDevice(activations, batch_size, activations_size);
  for(int i = 0; i < 2; i+=batch_size) {
    dotProductSegmented<<<nBlocks, nThreads>>>(d_input.get()+(i*model.layer_size[0]), model.d_weights[0], d_activations.get(), batch_size, model.layer_size[0], model.layer_size[0], model.layer_size[1], model.d_biases[0]);
    cudaDeviceSynchronize();
    sigmoidSegmented<<<nWorkers, nThreadsPerWorker>>>(d_activations.get(), batch_size*model.layer_size[1]);
    cudaDeviceSynchronize();
    dotProductSegmented<<<nBlocks, nThreads>>>(d_activations.get(), model.d_weights[1], d_activations.get()+(offsets[1]), batch_size, model.layer_size[1], model.layer_size[1], model.layer_size[2], model.d_biases[1]);
    cudaDeviceSynchronize();
    softmaxSegmented<<<nWorkers, nThreadsPerWorker>>>(d_activations.get()+(offsets[1]), batch_size, model.layer_size[2]);
    cudaDeviceSynchronize();
    cudaMemcpy(activations, d_activations.get(), activations_size*batch_size*sizeof(float), cudaMemcpyDeviceToHost);
    EXPECT_FLOAT_EQ(correctOutput[i], activations[2]);
  }
  free(offsets);
  free(weights[0]);
  free(weights[1]);
  free(biases[0]);
  free(biases[1]);
  free(weights);
  free(biases);
  free(activations);
}

TEST(ForwardPass, SingleThreadedSoftmaxEx1_BATCH_SIZE_2) {
  float correctOutput[2] = {1.0f, 1.0f};
  int nWorkers = 1;
  int nThreadsPerWorker = 1;
  dim3 nBlocks(nWorkers, 1, 1);
  dim3 nThreads(nThreadsPerWorker, 1, 1);
  int batch_size = 2;
  int nLayers = 2;
  float input[2] = {0.13000f, 0.42f};
  int *layers = new int[nLayers+1]{1, 2, 1};
  float **weights = new float*[2];
  weights[0] = new float[2]{0.1f, 0.2f};
  weights[1] = new float[2]{0.5f, 0.6f};
  float **biases = new float*[2];
  biases[0] = new float[2]{0.4f, 0.3f};
  biases[1] = new float[1]{0.7f};
  NeuralNetwork model(nLayers, layers, weights, biases, 1.0f);
  model.setupGPU(nWorkers*nThreadsPerWorker, batch_size);
  std::shared_ptr<float> d_input = transferMatrixToDevice(input, 2, model.layer_size[0]);
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
  float * activations = new float[batch_size*activations_size];
  //device pointers
  std::shared_ptr<int> d_offsets = transferMatrixToDevice(offsets, model.nLayers, 1);
  for(int i = 0; i < activations_size*batch_size; i++) {
    activations[i] = 1;
  }
  std::shared_ptr<float> d_activations = transferMatrixToDevice(activations, batch_size, activations_size);
  for(int i = 0; i < 2; i+=batch_size) {
    dotProductSegmented<<<nBlocks, nThreads>>>(d_input.get()+(i*model.layer_size[0]), model.d_weights[0], d_activations.get(), batch_size, model.layer_size[0], model.layer_size[0], model.layer_size[1], model.d_biases[0]);
    cudaDeviceSynchronize();
    sigmoidSegmented<<<nWorkers, nThreadsPerWorker>>>(d_activations.get(), batch_size*model.layer_size[1]);
    cudaDeviceSynchronize();
    dotProductSegmented<<<nBlocks, nThreads>>>(d_activations.get(), model.d_weights[1], d_activations.get()+(offsets[1]), batch_size, model.layer_size[1], model.layer_size[1], model.layer_size[2], model.d_biases[1]);
    cudaDeviceSynchronize();
    softmaxSegmented<<<nWorkers, nThreadsPerWorker>>>(d_activations.get()+(offsets[1]), batch_size, model.layer_size[2]);
    cudaDeviceSynchronize();
    cudaMemcpy(activations, d_activations.get(), activations_size*batch_size*sizeof(float), cudaMemcpyDeviceToHost);
    for(int j = 0; j < batch_size; j++) {
        EXPECT_FLOAT_EQ(correctOutput[j], activations[offsets[1]+j]);
    }
  }
  free(offsets);
  free(weights[0]);
  free(weights[1]);
  free(biases[0]);
  free(biases[1]);
  free(weights);
  free(biases);
  free(activations);
}

TEST(ForwardPass, SingleThreadedForwardPassEx2_BATCH_SIZE_1) {
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
  NeuralNetwork model(nLayers, layers, weights, biases, 1.0f);
  model.setupGPU(nWorkers*nThreadsPerWorker, batch_size);
  std::shared_ptr<float> d_input = transferMatrixToDevice(input, 2, model.layer_size[0]);
  int activations_size = 0;
  int * offsets = new int[model.nLayers];
  for(int i = 1; i <= model.nLayers; i++) {
    offsets[i-1] = (batch_size * activations_size);
    // printf("Offset at %d: %d\n", i-1, offsets[i-1]);
    activations_size += model.layer_size[i];
    
  }
  EXPECT_EQ(offsets[0], 0);
  EXPECT_EQ(offsets[1], model.layer_size[1]);
  EXPECT_EQ(offsets[2], offsets[1]+model.layer_size[2]);
  float * activations = new float[batch_size*activations_size];
  //device pointers
  std::shared_ptr<int> d_offsets = transferMatrixToDevice(offsets, model.nLayers, 1);
  for(int i = 0; i < activations_size*batch_size; i++) {
    activations[i] = 1;
  }
  std::shared_ptr<float> d_activations = transferMatrixToDevice(activations, batch_size, activations_size);
  dim3 nBlocks(nWorkers, 1, 1);
  dim3 nThreads(nThreadsPerWorker, 1, 1);
  for(int i = 0; i < 2; i+=batch_size) {
    dotProductSegmented<<<nBlocks, nThreads>>>(d_input.get()+(i*model.layer_size[0]), model.d_weights[0], d_activations.get(), batch_size, model.layer_size[0], model.layer_size[0], model.layer_size[1], model.d_biases[0]);
    cudaDeviceSynchronize();
    sigmoidSegmented<<<nWorkers, nThreadsPerWorker>>>(d_activations.get(), batch_size*model.layer_size[1]);
    cudaDeviceSynchronize();
    dotProductSegmented<<<nBlocks, nThreads>>>(d_activations.get(), model.d_weights[1], d_activations.get()+(offsets[1]), batch_size, model.layer_size[1], model.layer_size[1], model.layer_size[2], model.d_biases[1]);
    cudaDeviceSynchronize();
    sigmoidSegmented<<<nWorkers, nThreadsPerWorker>>>(d_activations.get()+offsets[1], batch_size*model.layer_size[2]);
    cudaDeviceSynchronize();
    dotProductSegmented<<<nWorkers, nThreadsPerWorker>>>(d_activations.get()+offsets[1], model.d_weights[2], d_activations.get()+offsets[2], batch_size, model.layer_size[2], model.layer_size[2], model.layer_size[3], model.d_biases[2]);
    cudaDeviceSynchronize();
    softmaxSegmented<<<nWorkers, nThreadsPerWorker>>>(d_activations.get()+(offsets[2]), batch_size, model.layer_size[3]);
    cudaMemcpy(activations, d_activations.get(), activations_size*batch_size*sizeof(float), cudaMemcpyDeviceToHost);
    printf("BATCH %d\n", i);
    for(int j = 0; j < activations_size*batch_size; j++) {
      printf("j: %d\n", i*activations_size+j);
      EXPECT_FLOAT_EQ(expected[i*activations_size+j], activations[j]);
    }
  }
  free(offsets);
  free(weights[0]);
  free(weights[1]);
  free(weights[2]);
  free(biases[0]);
  free(biases[1]);
  free(biases[2]);
  free(weights);
  free(biases);
  free(activations);
}

TEST(ForwardPass, SingleThreadedForwardPassEx2_BATCH_SIZE_2) {
  int nWorkers = 1;
  int nThreadsPerWorker = 1;
  dim3 nBlocks(nWorkers, 1, 1);
  dim3 nThreads(nThreadsPerWorker, 1, 1);
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
  NeuralNetwork model(nLayers, layers, weights, biases, 1.0f);
  model.setupGPU(nWorkers*nThreadsPerWorker, batch_size);
  std::shared_ptr<float> d_input = transferMatrixToDevice(input, 2, model.layer_size[0]);
  int activations_size = 0;
  int * offsets = new int[model.nLayers];
  for(int i = 1; i <= model.nLayers; i++) {
    offsets[i-1] = (batch_size * activations_size);
    // printf("Offset at %d: %d\n", i-1, offsets[i-1]);
    activations_size += model.layer_size[i];
    
  }
  EXPECT_EQ(offsets[0], 0);
  EXPECT_EQ(offsets[1], model.layer_size[1]*batch_size);
  EXPECT_EQ(offsets[2], offsets[1]+model.layer_size[2]*batch_size);
  float * activations = new float[batch_size*activations_size];
  //device pointers
  std::shared_ptr<int> d_offsets = transferMatrixToDevice(offsets, model.nLayers, 1);
  for(int i = 0; i < activations_size*batch_size; i++) {
    activations[i] = 1;
  }
  std::shared_ptr<float> d_activations = transferMatrixToDevice(activations, batch_size, activations_size);
  for(int i = 0; i < 2; i+=batch_size) {
    dotProductSegmented<<<nBlocks, nThreads>>>(d_input.get(), model.d_weights[0], d_activations.get(), batch_size, model.layer_size[0], model.layer_size[0], model.layer_size[1], model.d_biases[0]);
    cudaDeviceSynchronize();
    sigmoidSegmented<<<nWorkers, nThreadsPerWorker>>>(d_activations.get(), batch_size*model.layer_size[1]);
    cudaDeviceSynchronize();
    dotProductSegmented<<<nBlocks, nThreads>>>(d_activations.get(), model.d_weights[1], d_activations.get()+(offsets[1]), batch_size, model.layer_size[1], model.layer_size[1], model.layer_size[2], model.d_biases[1]);
    cudaDeviceSynchronize();
    sigmoidSegmented<<<nWorkers, nThreadsPerWorker>>>(d_activations.get()+offsets[1], batch_size*model.layer_size[2]);
    cudaDeviceSynchronize();
    dotProductSegmented<<<nWorkers, nThreadsPerWorker>>>(d_activations.get()+offsets[1], model.d_weights[2], d_activations.get()+offsets[2], batch_size, model.layer_size[2], model.layer_size[2], model.layer_size[3], model.d_biases[2]);
    cudaDeviceSynchronize();
    softmaxSegmented<<<nWorkers, nThreadsPerWorker>>>(d_activations.get()+(offsets[2]), batch_size, model.layer_size[3]);
    cudaMemcpy(activations, d_activations.get(), activations_size*batch_size*sizeof(float), cudaMemcpyDeviceToHost);
    for(int j = 0; j < activations_size*batch_size; j++) {
      EXPECT_FLOAT_EQ(expected[i*activations_size+j], activations[j]);
    }
  }
  free(offsets);
  free(weights[0]);
  free(weights[1]);
  free(weights[2]);
  free(biases[0]);
  free(biases[1]);
  free(biases[2]);
  free(weights);
  free(biases);
  free(activations);
}

