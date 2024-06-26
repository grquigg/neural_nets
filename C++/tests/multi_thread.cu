#include <gtest/gtest.h>
#include "../include/lin_alg.h"
#include "../include/utils.h"
#include "../include/models.h"

TEST(SegmentedDotProduct, DotProductMultiThreadedEx1) {
  int nWorkers = 1, nThreadsPerWorker = 1, batch_size = 2;
  dim3 nBlocks(nWorkers, 1, 1);
  dim3 nThreads(nThreadsPerWorker, 2, 1);
  int * layers = new int[2]{1,2};
  float correct[4] = {0.413f, 0.326f, 0.442f, 0.384f};
  float **weights = new float*[1]{new float[2]{0.1f, 0.2f}};
  float **biases = new float*[1]{new float[2]{0.4f, 0.3f}};
  NeuralNetwork model(1, layers, weights, biases, 1.0f);
  model.setupGPU(nWorkers*nThreadsPerWorker, batch_size);

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

TEST(SegmentedDotProduct, DotProductMultiThreadedEx2) {
  int nWorkers = 1;
  int nThreadsPerWorker = 1;
  dim3 nBlocks(nWorkers, 1, 1);
  dim3 nThreads(nThreadsPerWorker, 2, 1);
  int batch_size = 2;
  int layers[2] = {2, 4};
  float correct[8] = {0.74f, 1.1192f, 0.3564f, 0.8744f, 0.55250f, 0.81380f, 0.17610f, 0.60410f};
  float **weights = new float*[1]{new float[8]{0.15f, 0.1f, 0.19f, 0.35f, 0.4f, 0.54f, 0.42f, 0.68f}};
  float **biases = new float*[1]{new float[4]{0.42f, 0.72f, 0.01f, 0.3f}};
  NeuralNetwork model(1, layers, weights, biases, 1.0f);
  float input[4] = {0.32f, 0.68f, 0.83f, 0.02f};
  std::shared_ptr<float> d_weights = transferMatrixToDevice(model.weights[0], model.layer_size[0], model.layer_size[1]);
  std::shared_ptr<float> d_inputs = transferMatrixToDevice(input, 2, 2);
  std::shared_ptr<float> d_bias = transferMatrixToDevice(model.biases[0], model.layer_size[1], 1);
  float *d_product;
  cudaMalloc(&d_product, batch_size*model.layer_size[1]*sizeof(float));
  dotProductSegmented<<<nBlocks, nThreads>>>(d_inputs.get(), d_weights.get(), d_product, batch_size, model.layer_size[0], model.layer_size[0], model.layer_size[1], d_bias.get());
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
  dotProductSegmented<<<nBlocks, nThreads>>>(d_inputs.get(), d_weights.get(), d_product, batch_size, model.layer_size[0], model.layer_size[0], model.layer_size[1], d_bias.get());
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
  dotProductSegmented<<<nBlocks, nThreads>>>(d_inputs.get(), d_weights.get(), d_product, batch_size, model.layer_size[0], model.layer_size[0], model.layer_size[1], d_bias.get());
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
  dotProductSegmented<<<nBlocks, nThreads>>>(d_inputs.get(), d_weights.get(), d_product, batch_size, model.layer_size[0], model.layer_size[0], model.layer_size[1], d_bias.get());
  cudaDeviceSynchronize();
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

TEST(SegmentedSigmoid, MultiThreadedEx1) {
  int nWorkers = 1;
  int nThreadsPerWorker = 2;
  dim3 nBlocks(nWorkers, 1, 1);
  dim3 nThreads(nThreadsPerWorker, 1, 1);
  int batch_size = 2;
  int layers[2] = {1, 2};
  float correct[4] = {0.601807f, 0.58078581f, 0.6087355f, 0.59483749f};
  float **weights = new float*[1]{new float[2]{0.1f, 0.2f}};
  float **biases = new float*[1]{new float[2]{0.4f, 0.3f}};
  NeuralNetwork model(1, layers, weights, biases, 1.0f);
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

TEST(SegmentedSigmoid, MultiThreadedEx2) {
  int nWorkers = 2;
  int nThreadsPerWorker = 1;
  dim3 nBlocks(nWorkers, 1, 1);
  dim3 nThreads(nThreadsPerWorker, 1, 1);
  int batch_size = 2;
  int layers[2] = {2, 4};
  float correct[8] = {0.67699581f, 0.75384f, 0.58816868f, 0.7056604f, 0.63471538f, 0.69291866f, 0.54391158f, 0.64659375f};
  float **weights = new float*[1]{new float[8]{0.15f, 0.1f, 0.19f, 0.35f, 0.4f, 0.54f, 0.42f, 0.68f}};
  float **biases = new float*[1]{new float[4]{0.42f, 0.72f, 0.01f, 0.3f}};
  NeuralNetwork model(1, layers, weights, biases, 1.0f);
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
  free(weights[1]);
  free(biases[0]);
  free(biases[1]);
  free(weights);
  free(biases);
  free(activations);
}

TEST(ForwardPass, MultiThreadedSoftmaxEx1_BATCH_SIZE_1) {
  float correctOutput[2] = {1.0f, 1.0f};
  int nWorkers = 1;
  int nThreadsPerWorker = 1;
  dim3 nBlocks(nWorkers, 1, 1);
  dim3 nThreads(nThreadsPerWorker, 2, 1);
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
    nBlocks.y = 1;
    nThreads.y = 2;
    dotProductSegmented<<<nBlocks, nThreads>>>(d_input.get()+(i*model.layer_size[0]), model.d_weights[0], d_activations.get(), batch_size, model.layer_size[0], model.layer_size[0], model.layer_size[1], model.d_biases[0]);
    cudaDeviceSynchronize();
    sigmoidSegmented<<<nWorkers, nThreadsPerWorker>>>(d_activations.get(), batch_size*model.layer_size[1]);
    cudaDeviceSynchronize();
    nThreads.y = 1;
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

TEST(ForwardPass, MultiThreadedSoftmaxEx1_BATCH_SIZE_2) {
  float correctOutput[2] = {1.0f, 1.0f};
  int nWorkers = 2;
  int nThreadsPerWorker = 1;
  dim3 nBlocks(nWorkers, 1, 1);
  dim3 nThreads(nThreadsPerWorker, 2, 1);
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
  EXPECT_EQ(activations_size*batch_size, 6);;
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
    nThreads.y = 1;
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
  EXPECT_EQ(offsets[1], batch_size*model.layer_size[1]);
  EXPECT_EQ(offsets[2], offsets[1]+(batch_size*model.layer_size[2]));
  float * activations = new float[batch_size*activations_size];
  //device pointers
  std::shared_ptr<int> d_offsets = transferMatrixToDevice(offsets, model.nLayers, 1);
  for(int i = 0; i < activations_size*batch_size; i++) {
    activations[i] = 1;
  }
  std::shared_ptr<float> d_activations = transferMatrixToDevice(activations, batch_size, activations_size);
  dim3 nBlocks(nWorkers, 1, 1);
  dim3 nThreads(nThreadsPerWorker, 4, 1);
  for(int i = 0; i < 2; i+=batch_size) {
    dotProductSegmented<<<nBlocks, nThreads>>>(d_input.get()+(i*model.layer_size[0]), model.d_weights[0], d_activations.get(), batch_size, model.layer_size[0], model.layer_size[0], model.layer_size[1], model.d_biases[0]);
    cudaDeviceSynchronize();
    sigmoidSegmented<<<nWorkers, nThreadsPerWorker>>>(d_activations.get(), batch_size*model.layer_size[1]);
    cudaDeviceSynchronize();
    nThreads.y = 3;
    dotProductSegmented<<<nBlocks, nThreads>>>(d_activations.get(), model.d_weights[1], d_activations.get()+(offsets[1]), batch_size, model.layer_size[1], model.layer_size[1], model.layer_size[2], model.d_biases[1]);
    cudaDeviceSynchronize();
    sigmoidSegmented<<<nWorkers, nThreadsPerWorker>>>(d_activations.get()+offsets[1], batch_size*model.layer_size[2]);
    cudaDeviceSynchronize();
    nThreads.y = 2;
    dotProductSegmented<<<nBlocks, nThreads>>>(d_activations.get()+offsets[1], model.d_weights[2], d_activations.get()+offsets[2], batch_size, model.layer_size[2], model.layer_size[2], model.layer_size[3], model.d_biases[2]);
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

TEST(ForwardPass, MultiThreadedForwardPassEx2_BATCH_SIZE_2) {
  int nWorkers = 1;
  int nThreadsPerWorker = 1;
  dim3 nBlocks(nWorkers, 1, 1);
  dim3 nThreads(nThreadsPerWorker, 4, 1);
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
    dotProductSegmented<<<nBlocks, nThreads>>>(d_input.get()+(i*model.layer_size[0]), model.d_weights[0], d_activations.get(), batch_size, model.layer_size[0], model.layer_size[0], model.layer_size[1], model.d_biases[0]);
    cudaDeviceSynchronize();
    sigmoidSegmented<<<nWorkers, nThreadsPerWorker>>>(d_activations.get(), batch_size*model.layer_size[1]);
    cudaDeviceSynchronize();
    nThreads.y = 3;
    dotProductSegmented<<<nBlocks, nThreads>>>(d_activations.get(), model.d_weights[1], d_activations.get()+(offsets[1]), batch_size, model.layer_size[1], model.layer_size[1], model.layer_size[2], model.d_biases[1]);
    cudaDeviceSynchronize();
    sigmoidSegmented<<<nWorkers, nThreadsPerWorker>>>(d_activations.get()+offsets[1], batch_size*model.layer_size[2]);
    cudaDeviceSynchronize();
    nThreads.y = 2;
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

TEST(ForwardPass, MultiThreadedForwardPassEx2_BATCH_SIZE_2_1) {
  int nWorkers = 1;
  int nThreadsPerWorker = 1;
  dim3 nBlocks(nWorkers, 1, 1);
  dim3 nThreads(nThreadsPerWorker, 2, 1);
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
    dotProductSegmented<<<nBlocks, nThreads>>>(d_input.get()+(i*model.layer_size[0]), model.d_weights[0], d_activations.get(), batch_size, model.layer_size[0], model.layer_size[0], model.layer_size[1], model.d_biases[0]);
    cudaDeviceSynchronize();
    sigmoidSegmented<<<nWorkers, nThreadsPerWorker>>>(d_activations.get(), batch_size*model.layer_size[1]);
    cudaDeviceSynchronize();
    nThreads.y = 3;
    dotProductSegmented<<<nBlocks, nThreads>>>(d_activations.get(), model.d_weights[1], d_activations.get()+(offsets[1]), batch_size, model.layer_size[1], model.layer_size[1], model.layer_size[2], model.d_biases[1]);
    cudaDeviceSynchronize();
    sigmoidSegmented<<<nWorkers, nThreadsPerWorker>>>(d_activations.get()+offsets[1], batch_size*model.layer_size[2]);
    cudaDeviceSynchronize();
    nThreads.y = 2;
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

TEST(ForwardPass, NNForwardPass_Ex1) {
  float correctOutput[6] = {0.601807f, 0.58078581f, 0.6087355f, 0.59483749f, 1.0f, 1.0f};
  int nWorkers = 2;
  int nThreadsPerWorker = 1;
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
  NeuralNetwork model(nLayers, layers, weights, biases, 1.0);
  model.setupGPU(nWorkers*nThreadsPerWorker, batch_size);
  std::shared_ptr<float> d_input = transferMatrixToDevice(input, 2, 1);
  std::shared_ptr<float> activations = model.forward_pass(d_input, 2, batch_size, nWorkers, nThreadsPerWorker);
  for(int j = 0; j < 6; j++) {
    EXPECT_FLOAT_EQ(activations.get()[j], correctOutput[j]);
  }
}

TEST(ForwardPass, NNForwardPass_Ex2) {
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
  NeuralNetwork model(nLayers, layers, weights, biases, 1.0);
  model.setupGPU(nWorkers*nThreadsPerWorker, batch_size);
  std::shared_ptr<float> d_input = transferMatrixToDevice(input, 2, 2);
  std::shared_ptr<float> activations = model.forward_pass(d_input, 2, batch_size, nWorkers, nThreadsPerWorker);
  for(int j = 0; j < 18; j++) {
    EXPECT_FLOAT_EQ(activations.get()[j], expected[j]);
  }
}

TEST(ForwardPass, NNForwardPass_Ex1_1) {
  float correctOutput[3] = {0.601807f, 0.58078581f, 1.0f};
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
  NeuralNetwork model(nLayers, layers, weights, biases, 1.0);
  model.setupGPU(nWorkers*nThreadsPerWorker, batch_size);
  std::shared_ptr<float> d_input = transferMatrixToDevice(input, 2, 1);
  std::shared_ptr<float> activations = model.forward_pass(d_input, 2, batch_size, nWorkers, nThreadsPerWorker);
  for(int j = 0; j < 3; j++) {
    EXPECT_FLOAT_EQ(activations.get()[j], correctOutput[j]);
  }
}

TEST(ForwardPass, NNForwardPass_Ex1_2) {
  float correctOutput[3] = {0.60873549f, 0.59483749f, 1.0f};
  int nWorkers = 1;
  int nThreadsPerWorker = 1;
  int batch_size = 1;
  int nLayers = 2;
  float input[2] = {0.42f};
  int *layers = new int[nLayers+1]{1, 2, 1};
  float **weights = new float*[2];
  weights[0] = new float[2]{0.1f, 0.2f};
  weights[1] = new float[2]{0.5f, 0.6f};
  float **biases = new float*[2];
  biases[0] = new float[2]{0.4f, 0.3f};
  biases[1] = new float[1]{0.7f};
  NeuralNetwork model(nLayers, layers, weights, biases, 1.0);
  model.setupGPU(nWorkers*nThreadsPerWorker, batch_size);
  std::shared_ptr<float> d_input = transferMatrixToDevice(input, 1, 1);
  std::shared_ptr<float> activations = model.forward_pass(d_input, 2, batch_size, nWorkers, nThreadsPerWorker);
  for(int j = 0; j < 3; j++) {
    EXPECT_FLOAT_EQ(activations.get()[j], correctOutput[j]);
  }
}

TEST(CalculateDeltas, NNCalculateDeltasLayer1_Ex1) {
  float ys[2] = {0.9f, 0.23f};
  float correctDeltas[2] = {0.1f, 0.77f};
  int nWorkers = 2;
  int nThreadsPerWorker = 1;
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
  NeuralNetwork model(nLayers, layers, weights, biases, 1.0);
  model.setupGPU(nWorkers*nThreadsPerWorker, batch_size);
  std::shared_ptr<float> d_input = transferMatrixToDevice(input, 2, 1);
  model.forward_pass(d_input, 2, batch_size, nWorkers, nThreadsPerWorker);
  
  //compute deltas
  float* d_deltas;
  std::shared_ptr<float> d_y = transferMatrixToDevice(ys, 1, 2);
  std::cout << model.layer_size[model.nLayers] << std::endl;
  cudaMalloc(&d_deltas, batch_size*model.layer_size[model.nLayers]*sizeof(float));
  matrixSubtract<<<model.layer_size[model.nLayers],batch_size>>>(model.activations+model.offsets[model.nLayers-1], d_y.get(), model.layer_size[model.nLayers], batch_size, model.layer_size[model.nLayers], batch_size, d_deltas);
  float * deltas = new float[batch_size*model.layer_size[model.nLayers]];
  cudaMemcpy(deltas, d_deltas, batch_size*model.layer_size[model.nLayers]*sizeof(float), cudaMemcpyDeviceToHost);
  for(int i = 0; i < batch_size*model.layer_size[model.nLayers]; i++) {
    EXPECT_FLOAT_EQ(correctDeltas[i], deltas[i]);
  }
}

TEST(CalculateDeltas, NNCalculateDeltasLayer2_Ex2) {
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
  float ys[4] = {0.75f, 0.98f, 0.75f, 0.28f};
  float correctDeltas[4] = {-0.2649302f ,-0.4650698f, -0.2658681f, 0.2358681f};
  NeuralNetwork model(nLayers, layers, weights, biases, 1.0);
  model.setupGPU(nWorkers*nThreadsPerWorker, batch_size);
  std::shared_ptr<float> d_input = transferMatrixToDevice(input, 2, 2);
  model.forward_pass(d_input, 2, batch_size, nWorkers, nThreadsPerWorker);
  float* d_deltas;
  cudaMalloc(&d_deltas, batch_size*model.layer_size[model.nLayers]*sizeof(float));
  std::shared_ptr<float> d_y = transferMatrixToDevice(ys, 2, 2);
  matrixSubtract<<<model.layer_size[model.nLayers],batch_size>>>(model.activations+model.offsets[model.nLayers-1], d_y.get(), model.layer_size[model.nLayers], batch_size, model.layer_size[model.nLayers], batch_size, d_deltas);
  float * deltas = new float[batch_size*model.layer_size[model.nLayers]];
  cudaMemcpy(deltas, d_deltas, batch_size*model.layer_size[model.nLayers]*sizeof(float), cudaMemcpyDeviceToHost);
  for(int i = 0; i < batch_size*model.layer_size[model.nLayers]; i++) {
    EXPECT_FLOAT_EQ(correctDeltas[i], deltas[i]);
  }
}

TEST(CalculateDeltas, NNCalculateDeltasLayer0_Ex1) {
  float ys[2] = {0.9f, 0.23f};
  float correctDeltas[4] = {0.05f, 0.060000017f, 0.385f, 0.462f};
  int nWorkers = 2;
  int nThreadsPerWorker = 1;
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
  NeuralNetwork model(nLayers, layers, weights, biases, 1.0);
  model.setupGPU(nWorkers*nThreadsPerWorker, batch_size);

  std::shared_ptr<float> d_input = transferMatrixToDevice(input, 2, 1);
  model.forward_pass(d_input, 2, batch_size, nWorkers, nThreadsPerWorker);
  //compute deltas
  float* d_deltas;
  std::shared_ptr<float> d_y = transferMatrixToDevice(ys, 1, 2);
  std::cout << model.layer_size[model.nLayers] << std::endl;
  cudaMalloc(&d_deltas, batch_size*model.layer_size[model.nLayers]*sizeof(float));
  matrixSubtract<<<model.layer_size[model.nLayers],batch_size>>>(model.activations+model.offsets[model.nLayers-1], d_y.get(), batch_size, model.layer_size[model.nLayers], batch_size, model.layer_size[model.nLayers], d_deltas);
  float* d_deltas0;
  cudaMalloc(&d_deltas0, batch_size*model.layer_size[model.nLayers-1]*sizeof(float));
  std::cout << batch_size << " " << model.layer_size[model.nLayers-1] << std::endl;
  //there's ambiguity in the dotProductTransposeSegmented
  dotProductTransposeSegmented<<<batch_size,model.nLayers-1>>>(d_deltas, model.d_weights[model.nLayers-1], d_deltas0, batch_size, model.layer_size[model.nLayers], model.layer_size[model.nLayers-1], model.layer_size[model.nLayers], false);
  float *deltas = new float[batch_size*model.layer_size[model.nLayers-1]*sizeof(float)];
  cudaMemcpy(deltas, d_deltas0, batch_size*model.layer_size[model.nLayers-1]*sizeof(float), cudaMemcpyDeviceToHost);
  for(int i = 0; i < batch_size*model.layer_size[model.nLayers-1]; i++) {
    EXPECT_FLOAT_EQ(correctDeltas[i], deltas[i]);
  }
}

TEST(CalculateDeltas, NNCalculateDeltasLayer1_Ex2) {
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
  float ys[4] = {0.75f, 0.98f, 0.75f, 0.28f};
  float correctDeltas[6] = {-0.27699625f, -0.55308699f, -0.46131117f, -0.20771844f,0.11241009f,0.021838875f};
  NeuralNetwork model(nLayers, layers, weights, biases, 1.0);
  model.setupGPU(nWorkers*nThreadsPerWorker, batch_size);

  std::shared_ptr<float> d_input = transferMatrixToDevice(input, 2, 2);
  model.forward_pass(d_input, 2, batch_size, nWorkers, nThreadsPerWorker);
  float* d_deltas;
  cudaMalloc(&d_deltas, batch_size*model.layer_size[model.nLayers]*sizeof(float));
  std::shared_ptr<float> d_y = transferMatrixToDevice(ys, 2, 2);
  matrixSubtract<<<batch_size,model.layer_size[model.nLayers]>>>(model.activations+model.offsets[model.nLayers-1], d_y.get(), batch_size, model.layer_size[model.nLayers], batch_size, model.layer_size[model.nLayers], d_deltas);
  float* d_deltas0;
  cudaMalloc(&d_deltas0, batch_size*model.layer_size[model.nLayers-1]*sizeof(float));
  dotProductTransposeSegmented<<<batch_size, model.layer_size[model.nLayers-1]>>>(d_deltas, model.d_weights[model.nLayers-1], d_deltas0, batch_size, model.layer_size[model.nLayers], model.layer_size[model.nLayers-1], model.layer_size[model.nLayers], false);
  float *deltas = new float[batch_size*model.layer_size[model.nLayers-1]*sizeof(float)];
  cudaMemcpy(deltas, d_deltas0, batch_size*model.layer_size[model.nLayers-1]*sizeof(float), cudaMemcpyDeviceToHost);
  for(int i = 0; i < batch_size*model.layer_size[model.nLayers-1]; i++) {
    EXPECT_FLOAT_EQ(correctDeltas[i], deltas[i]);
  }
}

TEST(SigmoidDerivative, NNCalculateDeltasLayer1_Ex2) {
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
  float ys[4] = {0.75f, 0.98f, 0.75f, 0.28f};
  float correctDeltas[6] = {-0.03025601f, -0.05286462f, -0.06961101f,-0.02497924f,0.011581795f,0.0035215414f};
  NeuralNetwork model(nLayers, layers, weights, biases, 1.0);
  model.setupGPU(nWorkers*nThreadsPerWorker, batch_size);

  std::shared_ptr<float> d_input = transferMatrixToDevice(input, 2, 2);
  model.forward_pass(d_input, 2, batch_size, nWorkers, nThreadsPerWorker);
  float* d_deltas;
  cudaMalloc(&d_deltas, batch_size*model.layer_size[model.nLayers]*sizeof(float));
  std::shared_ptr<float> d_y = transferMatrixToDevice(ys, 2, 2);
  matrixSubtract<<<batch_size,model.layer_size[model.nLayers]>>>(model.activations+model.offsets[model.nLayers-1], d_y.get(), batch_size, model.layer_size[model.nLayers], batch_size, model.layer_size[model.nLayers], d_deltas);
  float* d_deltas0;
  cudaMalloc(&d_deltas0, batch_size*model.layer_size[model.nLayers-1]*sizeof(float));
  dotProductTransposeSegmented<<<batch_size, model.layer_size[model.nLayers-1]>>>(d_deltas, model.d_weights[model.nLayers-1], d_deltas0, batch_size, model.layer_size[model.nLayers], model.layer_size[model.nLayers-1], model.layer_size[model.nLayers], false);
  cudaDeviceSynchronize();
  sigmoidD<<<batch_size, model.layer_size[model.nLayers-1]>>>(model.activations+model.offsets[model.nLayers-2], batch_size, model.layer_size[model.nLayers-1], d_deltas0);
  cudaDeviceSynchronize();
  float *deltas = new float[batch_size*model.layer_size[model.nLayers-1]*sizeof(float)];
  cudaMemcpy(deltas, d_deltas0, batch_size*model.layer_size[model.nLayers-1]*sizeof(float), cudaMemcpyDeviceToHost);
  for(int i = 0; i < batch_size*model.layer_size[model.nLayers-1]; i++) {
    EXPECT_FLOAT_EQ(correctDeltas[i], deltas[i]);
  }
}

TEST(SigmoidDerivative, NNCalculateDeltasLayer0_Ex2) {
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
  float ys[4] = {0.75f, 0.98f, 0.75f, 0.28f};
  float correctDeltas[8] = {-0.01781237f, -0.01308189f, -0.022767831f, -0.01654095f,-0.0022952568f,0.00034821813f,-0.0044266f,-0.00253811f};
  NeuralNetwork model(nLayers, layers, weights, biases, 1.0);
  model.setupGPU(nWorkers*nThreadsPerWorker, batch_size);

  std::shared_ptr<float> d_input = transferMatrixToDevice(input, 2, 2);
  model.forward_pass(d_input, 2, batch_size, nWorkers, nThreadsPerWorker);
  float* d_deltas;
  cudaMalloc(&d_deltas, batch_size*model.layer_size[model.nLayers]*sizeof(float));
  std::shared_ptr<float> d_y = transferMatrixToDevice(ys, 2, 2);
  matrixSubtract<<<batch_size,model.layer_size[model.nLayers]>>>(model.activations+model.offsets[model.nLayers-1], d_y.get(), batch_size, model.layer_size[model.nLayers], batch_size, model.layer_size[model.nLayers], d_deltas);
  float* d_deltas0;
  cudaMalloc(&d_deltas0, batch_size*model.layer_size[model.nLayers-1]*sizeof(float));
  dotProductTransposeSegmented<<<batch_size, model.layer_size[model.nLayers-1]>>>(d_deltas, model.d_weights[model.nLayers-1], d_deltas0, batch_size, model.layer_size[model.nLayers], model.layer_size[model.nLayers-1], model.layer_size[model.nLayers], false);
  cudaDeviceSynchronize();
  sigmoidD<<<batch_size, model.layer_size[model.nLayers-1]>>>(model.activations+model.offsets[model.nLayers-2], batch_size, model.layer_size[model.nLayers-1], d_deltas0);
  cudaDeviceSynchronize();
  float* d_deltas1;
  cudaMalloc(&d_deltas1, batch_size*model.layer_size[model.nLayers-2]*sizeof(float));
  dotProductTransposeSegmented<<<batch_size, model.layer_size[model.nLayers-2]>>>(d_deltas0, model.d_weights[model.nLayers-2], d_deltas1, batch_size, model.layer_size[model.nLayers-1], model.layer_size[model.nLayers-2], model.layer_size[model.nLayers-1], false);
  cudaDeviceSynchronize();
  sigmoidD<<<batch_size, model.layer_size[model.nLayers-2]>>>(model.activations+model.offsets[model.nLayers-3], batch_size, model.layer_size[model.nLayers-2], d_deltas1);
  cudaDeviceSynchronize();
  float *deltas = new float[batch_size*model.layer_size[model.nLayers-2]*sizeof(float)];
  cudaMemcpy(deltas, d_deltas1, batch_size*model.layer_size[model.nLayers-2]*sizeof(float), cudaMemcpyDeviceToHost);
  for(int i = 0; i < batch_size*model.layer_size[model.nLayers-2]; i++) {
    EXPECT_FLOAT_EQ(correctDeltas[i], deltas[i]);
  }
};

TEST(CalculateDeltas, NNCalculateDeltasLayer0_Ex2) {
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
  float ys[4] = {0.75f, 0.98f, 0.75f, 0.28f};
  float correctDeltas[8] = {-0.08145683f, -0.07049757f, -0.09399404f, -0.07963723f,-0.00989967f, 0.0016364987f,-0.01784403f, -0.0111072f};
  NeuralNetwork model(nLayers, layers, weights, biases, 1.0);
  model.setupGPU(nWorkers*nThreadsPerWorker, batch_size);

  std::shared_ptr<float> d_input = transferMatrixToDevice(input, 2, 2);
  model.forward_pass(d_input, 2, batch_size, nWorkers, nThreadsPerWorker);
  float* d_deltas;
  cudaMalloc(&d_deltas, batch_size*model.layer_size[model.nLayers]*sizeof(float));
  std::shared_ptr<float> d_y = transferMatrixToDevice(ys, 2, 2);
  matrixSubtract<<<batch_size,model.layer_size[model.nLayers]>>>(model.activations+model.offsets[model.nLayers-1], d_y.get(), batch_size, model.layer_size[model.nLayers], batch_size, model.layer_size[model.nLayers], d_deltas);
  float* d_deltas0;
  cudaMalloc(&d_deltas0, batch_size*model.layer_size[model.nLayers-1]*sizeof(float));
  dotProductTransposeSegmented<<<batch_size, model.layer_size[model.nLayers-1]>>>(d_deltas, model.d_weights[model.nLayers-1], d_deltas0, batch_size, model.layer_size[model.nLayers], model.layer_size[model.nLayers-1], model.layer_size[model.nLayers], false);
  cudaDeviceSynchronize();
  sigmoidD<<<batch_size, model.layer_size[model.nLayers-1]>>>(model.activations+model.offsets[model.nLayers-2], batch_size, model.layer_size[model.nLayers-1], d_deltas0);
  cudaDeviceSynchronize();
  float* d_deltas1;
  cudaMalloc(&d_deltas1, batch_size*model.layer_size[model.nLayers-2]*sizeof(float));
  dotProductTransposeSegmented<<<batch_size, model.layer_size[model.nLayers-2]>>>(d_deltas0, model.d_weights[model.nLayers-2], d_deltas1, batch_size, model.layer_size[model.nLayers-1], model.layer_size[model.nLayers-2], model.layer_size[model.nLayers-1], false);

  float *deltas = new float[batch_size*model.layer_size[model.nLayers-2]*sizeof(float)];
  cudaMemcpy(deltas, d_deltas1, batch_size*model.layer_size[model.nLayers-2]*sizeof(float), cudaMemcpyDeviceToHost);
  for(int i = 0; i < batch_size*model.layer_size[model.nLayers-2]; i++) {
    EXPECT_FLOAT_EQ(correctDeltas[i], deltas[i]);
  }
}

TEST(CalculateDeltas, NNExample1) { 
  /*this is mostly just to test that the backprop code 
  implemented works the way that we expect it to */
  int nWorkers = 2;
  int nThreadsPerWorker = 1;
  int batch_size = 2;
  int nLayers = 2;
  float input[2] = {0.13000f, 0.42f};
  float ys[2] = {0.9f, 0.23f};
  int *layers = new int[nLayers+1]{1, 2, 1};
  float **weights = new float*[2];
  weights[0] = new float[2]{0.1f, 0.2f};
  weights[1] = new float[2]{0.5f, 0.6f};
  float **biases = new float*[2];
  biases[0] = new float[2]{0.4f, 0.3f};
  biases[1] = new float[1]{0.7f};
  NeuralNetwork model(nLayers, layers, weights, biases, 1.0);
  model.setupGPU(nWorkers*nThreadsPerWorker, batch_size);
  std::shared_ptr<float> d_input = transferMatrixToDevice(input, 2, 1);
  std::shared_ptr<float> d_y = transferMatrixToDevice(ys, 1, 2);
  std::shared_ptr<float> activations = model.forward_pass(d_input, 2, batch_size, nWorkers, nThreadsPerWorker);
  
  model.backprop(batch_size, d_input, d_y);
  float **deltas = new float*[model.nLayers];
  for(int i = 0; i < model.nLayers; i++) {
    std::cout << "Batch: " << batch_size*model.layer_size[i+1] << std::endl;
    deltas[i] = new float[batch_size*model.layer_size[i+1]];
    cudaMemcpy(deltas[i], model.deltas[i], batch_size*model.layer_size[i+1]*sizeof(float), cudaMemcpyDeviceToHost);
  }
  std::cout << "Here" << std::endl;
  float correctDeltas1[2] = {0.1f, 0.77f};
  float correctDeltas0[4] = {0.011981769f, 0.01460842f, 0.09169799f, 0.1113447f};
  for(int j = 0; j < 2; j++) {
    EXPECT_FLOAT_EQ(deltas[model.nLayers-1][j], correctDeltas1[j]);
  }
  for(int j = 0; j < 4; j++) {
    EXPECT_FLOAT_EQ(correctDeltas0[j], deltas[0][j]);
  }
}

TEST(CalculateDeltas, NNExample2) {
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
  float ys[4] = {0.75f, 0.98f, 0.75f, 0.28f};
  NeuralNetwork model(nLayers, layers, weights, biases, 1.0);
  model.setupGPU(nWorkers*nThreadsPerWorker, batch_size);
  std::shared_ptr<float> d_input = transferMatrixToDevice(input, 2, 2);
  std::shared_ptr<float> d_y = transferMatrixToDevice(ys, 2, 2);
  model.forward_pass(d_input, 2, batch_size, nWorkers, nThreadsPerWorker);
  
  model.backprop(batch_size, d_input, d_y);
  float **deltas = new float*[model.nLayers];
  float **correctDeltas = new float*[model.nLayers];
  correctDeltas[2] = new float[4]{-0.2649302f ,-0.4650698f, -0.2658681f, 0.2358681f};
  correctDeltas[1] = new float[6]{-0.03025601f, -0.05286462f, -0.06961101f,-0.02497924f,0.011581795f,0.0035215414f};
  correctDeltas[0] = new float[8]{-0.01781237f, -0.01308189f, -0.022767831f, -0.01654095f,-0.0022952568f,0.00034821813f,-0.0044266f,-0.00253811f};
  for(int i = 0; i < model.nLayers; i++) {
    deltas[i] = new float[batch_size*model.layer_size[i+1]];
    cudaMemcpy(deltas[i], model.deltas[i], batch_size*model.layer_size[i+1]*sizeof(float), cudaMemcpyDeviceToHost);
    for(int j = 0; j < batch_size*model.layer_size[i+1]; j++) {
      EXPECT_FLOAT_EQ(correctDeltas[i][j], deltas[i][j]);
    }
  }
}

TEST(ComputeGradients, NNExample1Gradient) {
  int nWorkers = 2;
  int nThreadsPerWorker = 1;
  int batch_size = 2;
  int nLayers = 2;
  float input[2] = {0.13000f, 0.42f};
  float ys[2] = {0.9f, 0.23f};
  int *layers = new int[nLayers+1]{1, 2, 1};
  float **weights = new float*[2];
  weights[0] = new float[2]{0.1f, 0.2f};
  weights[1] = new float[2]{0.5f, 0.6f};
  float **biases = new float*[2];
  biases[0] = new float[2]{0.4f, 0.3f};
  biases[1] = new float[1]{0.7f};
  NeuralNetwork model(nLayers, layers, weights, biases, 1.0);
  model.setupGPU(nWorkers*nThreadsPerWorker, batch_size);
  std::shared_ptr<float> d_input = transferMatrixToDevice(input, 2, 1);
  std::shared_ptr<float> d_y = transferMatrixToDevice(ys, 1, 2);
  std::shared_ptr<float> activations = model.forward_pass(d_input, 2, batch_size, nWorkers, nThreadsPerWorker);
  
  model.backprop(batch_size, d_input, d_y);
  float **expected_grads = new float*[model.nLayers];
  expected_grads[1] = new float[model.layer_size[1]*model.layer_size[2]]{0.52890703f, 0.51610345f};
  expected_grads[0] = new float[model.layer_size[0]*model.layer_size[1]]{0.04007078f, 0.04866387f};
  float **gradients = new float*[model.nLayers];
  for(int i = 0; i < model.nLayers; i++) {
    gradients[i] = new float[model.layer_size[i]*model.layer_size[i+1]];
    cudaMemcpy(gradients[i], model.gradients[i], model.layer_size[i]*model.layer_size[i+1]*sizeof(float), cudaMemcpyDeviceToHost);
    for(int j = 0; j < model.layer_size[i]*model.layer_size[i+1]; j++) {
      EXPECT_FLOAT_EQ(expected_grads[i][j], gradients[i][j]);
    }
  }
}

TEST(ComputeGradients, NNExample2Gradient) {
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
  float ys[4] = {0.75f, 0.98f, 0.75f, 0.28f};
  NeuralNetwork model(nLayers, layers, weights, biases, 1.0);
  model.setupGPU(nWorkers*nThreadsPerWorker, batch_size);
  std::shared_ptr<float> d_input = transferMatrixToDevice(input, 2, 2);
  std::shared_ptr<float> d_y = transferMatrixToDevice(ys, 2, 2);
  model.forward_pass(d_input, 2, batch_size, nWorkers, nThreadsPerWorker);
  
  model.backprop(batch_size, d_input, d_y);
  float **deltas = new float*[model.nLayers];
  float **correctDeltas = new float*[model.nLayers];
  correctDeltas[2] = new float[4]{-0.2649302f ,-0.4650698f, -0.2658681f, 0.2358681f};
  correctDeltas[1] = new float[6]{-0.03025601f, -0.05286462f, -0.06961101f,-0.02497924f,0.011581795f,0.0035215414f};
  correctDeltas[0] = new float[8]{-0.01781237f, -0.01308189f, -0.022767831f, -0.01654095f,-0.0022952568f,0.00034821813f,-0.0044266f,-0.00253811f};
  for(int i = 0; i < model.nLayers; i++) {
    deltas[i] = new float[batch_size*model.layer_size[i+1]];
    cudaMemcpy(deltas[i], model.deltas[i], batch_size*model.layer_size[i+1]*sizeof(float), cudaMemcpyDeviceToHost);
    for(int j = 0; j < batch_size*model.layer_size[i+1]; j++) {
      EXPECT_FLOAT_EQ(correctDeltas[i][j], deltas[i][j]);
    }
  }
}