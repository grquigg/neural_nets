#include <gtest/gtest.h>
#include "../include/lin_alg.h"
#include "../include/utils.h"
#include "../include/models.h"
// Demonstrate some basic assertions.

TEST(Main, TestBuildModel) { 
  int layers[2] = {1, 2};
  float **weights = new float*[1]{new float[2]{0.1f, 0.2f}};
  float **biases = new float*[1]{new float[2]{0.4f, 0.3f}};
  NeuralNetwork model(1, layers, weights, biases, 1.0f);
  EXPECT_EQ(model.weights[0][0], 0.1f);
  EXPECT_EQ(model.weights[0][1], 0.2f);
  EXPECT_EQ(model.biases[0][0], 0.4f);
  EXPECT_EQ(model.biases[0][1], 0.3f);
  EXPECT_EQ(model.nLayers, 1);
  EXPECT_EQ(model.layer_size[0], 1);
  EXPECT_EQ(model.layer_size[1], 2);
}

TEST(Main, TestCopyModelToGPU) {
  int nWorkers = 1, nThreadsPerWorker = 1;
  int layers[2] = {1, 2};
  float **weights = new float*[1]{new float[2]{0.1f, 0.2f}};
  float **biases = new float*[1]{new float[2]{0.4f, 0.3f}};
  NeuralNetwork model(1, layers, weights, biases, 1.0f);
  model.on_device = true;
  std::shared_ptr<NeuralNetwork> d_model = copyModelToGPU(&model, nWorkers, nThreadsPerWorker);
  NeuralNetwork *temp = new NeuralNetwork;
  temp->weights = new float*[1];
  temp->biases = new float*[1];
  temp->layer_size = new int[2];
  float * temp_weights = new float[2];
  float * temp_biases = new float[2];
  cudaMemcpy(temp_weights, d_model->weights[0], sizeof(float*), cudaMemcpyDeviceToHost);
  cudaMemcpy(temp_biases, d_model->biases[0], sizeof(float*), cudaMemcpyDeviceToHost);
  cudaMemcpy(temp->layer_size, d_model->layer_size, 2*sizeof(int), cudaMemcpyDeviceToHost);
  EXPECT_FLOAT_EQ(temp_weights[0], 0.1f);
  EXPECT_FLOAT_EQ(temp_weights[1], 0.2f);
  EXPECT_EQ(temp->layer_size[0], 1);
  EXPECT_EQ(temp->layer_size[1], 2);
  EXPECT_FLOAT_EQ(temp_biases[0], 0.4f);
  EXPECT_FLOAT_EQ(temp_biases[1], 0.3f);
  printf("Success\n");

}
