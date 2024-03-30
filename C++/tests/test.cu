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
  free(weights[0]);
  free(weights);
  free(biases[0]);
  free(biases);
}

TEST(Main, TestCopyModelToGPU) {
  int nWorkers = 1, nThreadsPerWorker = 1;
  int layers[2] = {1, 2};
  float **weights = new float*[1]{new float[2]{0.1f, 0.2f}};
  float **biases = new float*[1]{new float[2]{0.4f, 0.3f}};
  NeuralNetwork model(1, layers, weights, biases, 1.0f);
  model.on_device = true;
  std::shared_ptr<NeuralNetwork> d_model = copyModelToGPU(&model, nWorkers, nThreadsPerWorker);
  std::shared_ptr<NeuralNetwork> temp = std::make_shared<NeuralNetwork>();
  temp->weights = new float*[1];
  temp->biases = new float*[1];
  temp.get()->layer_size = new int[2];
  float * temp_weights = new float[2];
  float * temp_biases = new float[2];
  printf("%d\n", temp->layer_size[0]);
  cudaMemcpy(temp_weights, d_model->weights[0], sizeof(float*), cudaMemcpyDeviceToHost);
  cudaMemcpy(temp_biases, d_model->biases[0], sizeof(float*), cudaMemcpyDeviceToHost);
  EXPECT_FLOAT_EQ(temp_weights[0], 0.1f);
  EXPECT_FLOAT_EQ(temp_weights[1], 0.2f);
  EXPECT_FLOAT_EQ(temp_biases[0], 0.4f);
  EXPECT_FLOAT_EQ(temp_biases[1], 0.3f);
  free(temp_biases);
  free(temp_weights);
  free(weights[0]);
  free(weights);
  free(biases[0]);
  free(biases);

}

TEST(Main, MultiThreadedSubtraction) {
  float input[64];
  float product[64];
  for(int i = 0; i < 64; i++) {
    input[i] = 1.0f;
    product[i] = -i;
  }
  std::shared_ptr<float> d_input = transferMatrixToDevice(input, 8, 8);
  std::shared_ptr<float> d_product = transferMatrixToDevice(product, 8, 8);
  //this should just call the matrixSubtract method from lin_alg.cu directly
  matrixSubtract<<<8,8>>>(d_input.get(), d_product.get(), 8, 8, 8, 8, d_product.get());
  cudaMemcpy(product, d_product.get(), 64*sizeof(float), cudaMemcpyDeviceToHost);
  for(int i = 0; i < 64; i++) {
    EXPECT_FLOAT_EQ(product[i], 1.0f+i);
  }
}