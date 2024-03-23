#include <gtest/gtest.h>
#include "../include/lin_alg.h"
#include "../include/utils.h"
#include "../include/models.h"
// Demonstrate some basic assertions.

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
