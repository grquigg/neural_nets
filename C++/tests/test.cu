#include <gtest/gtest.h>
#include "../include/lin_alg.h"
#include "../include/utils.h"
#include "../include/models.h"
// Demonstrate some basic assertions.
TEST(SegmentedDotProduct, SingleThreaded) {
    int nWorkers = 1, nThreadsPerWorker = 1;
    dim3 nBlocks(nWorkers, 1, 1);
    dim3 nThreads(nThreadsPerWorker, 1, 1);
    float arr1[6] = {1,2,3,4,5,6};
    float arr2[12] = {-1,-2,-3,-4,-5,-6,-7,-8,-9,-10,-11,-12};
    float product[8];
    float correct_ans[8] = {-38.0f, -44.0f, -50.0f, -56.0f, -83.0f, -98.0f, -113.0f, -128.0f};
    std::shared_ptr<float> darr1 = transferMatrixToDevice(arr1, 2, 3);
    std::shared_ptr<float> darr2 = transferMatrixToDevice(arr2, 3, 4);
    float *dproduct;
    cudaMalloc(&dproduct, 8*sizeof(float));
    dotProductSegmented<<<nBlocks, nThreads>>>(darr1.get(), darr2.get(), dproduct, 2, 3, 3, 4);
    cudaDeviceSynchronize();
    cudaMemcpy(product, dproduct, 8*sizeof(float), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    for(int i = 0; i < 2; i++) {
        for(int j = 0; j < 4; j++) {
            EXPECT_EQ(product[i*4+j], correct_ans[i*4+j]);
        }
    }
    cudaFree(dproduct);
}

TEST(SegmentedDotProduct, MultiThreaded) {
    int nWorkers = 1, nThreadsPerWorker = 1;
    dim3 nBlocks(nWorkers, 2, 1);
    dim3 nThreads(nThreadsPerWorker, 2, 1);
    float arr1[6] = {1,2,3,4,5,6};
    float arr2[12] = {-1,-2,-3,-4,-5,-6,-7,-8,-9,-10,-11,-12};
    float product[8];
    float correct_ans[8] = {-38.0f, -44.0f, -50.0f, -56.0f, -83.0f, -98.0f, -113.0f, -128.0f};
    std::shared_ptr<float> darr1 = transferMatrixToDevice(arr1, 2, 3);
    std::shared_ptr<float> darr2 = transferMatrixToDevice(arr2, 3, 4);
    float *dproduct;
    cudaMalloc(&dproduct, 8*sizeof(float));
    dotProductSegmented<<<nBlocks, nThreads>>>(darr1.get(), darr2.get(), dproduct, 2, 3, 3, 4);
    cudaDeviceSynchronize();
    cudaMemcpy(product, dproduct, 8*sizeof(float), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    for(int i = 0; i < 2; i++) {
        for(int j = 0; j < 4; j++) {
            EXPECT_EQ(product[i*4+j], correct_ans[i*4+j]);
        }
    }
    cudaFree(dproduct);
}


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
  model.setupGPU(nWorkers*nThreadsPerWorker);
  std::shared_ptr<NeuralNetwork> temp = std::make_shared<NeuralNetwork>();
  temp->weights = new float*[1];
  temp->biases = new float*[1];
  temp.get()->layer_size = new int[2];
  float * temp_weights = new float[2];
  float * temp_biases = new float[2];
  printf("%d\n", temp->layer_size[0]);
  cudaMemcpy(temp_weights, model.d_weights[0], sizeof(float*), cudaMemcpyDeviceToHost);
  cudaMemcpy(temp_biases, model.d_biases[0], sizeof(float*), cudaMemcpyDeviceToHost);
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

TEST(Main, MultiThreadedDotProductTransposeMatrix1) {
    int nWorkers = 1, nThreadsPerWorker = 1;
    // dim3 nBlocks(nWorkers, 2, 1); //deals with the columns in the first matrix
    // dim3 nThreads(nThreadsPerWorker, 2, 1); //deals with the column in the second matrix
    float arr1[6] = {1,4,2,5,3,6}; // 3 x 2 matrix
    float arr2[12] = {-1,-2,-3,-4,-5,-6,-7,-8,-9,-10,-11,-12}; // 3 x 4 matrix
    float product[8];
    float correct_ans[8] = {-38.0f, -44.0f, -50.0f, -56.0f, -83.0f, -98.0f, -113.0f, -128.0f};
    std::shared_ptr<float> darr1 = transferMatrixToDevice(arr1, 3, 2);
    std::shared_ptr<float> darr2 = transferMatrixToDevice(arr2, 3, 4);
    float *dproduct;
    cudaMalloc(&dproduct, 8*sizeof(float));
    dotProductTransposeSegmented<<<1, 1>>>(darr1.get(), darr2.get(), dproduct, 3, 2, 3, 4, true);
    cudaDeviceSynchronize();
    cudaMemcpy(product, dproduct, 8*sizeof(float), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    for(int i = 0; i < 8; i++) {
        EXPECT_EQ(product[i], correct_ans[i]);
        product[i] = 0.0f;
    }
    dotProductTransposeSegmented<<<2,2>>>(darr1.get(), darr2.get(), dproduct, 3, 2, 3, 4, true);
    cudaDeviceSynchronize();
    cudaMemcpy(product, dproduct, 8*sizeof(float), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    for(int i = 0; i < 8; i++) {
        EXPECT_EQ(product[i], correct_ans[i]);
        product[i] = 0.0f;
    }
    dotProductTransposeSegmented<<<2,4>>>(darr1.get(), darr2.get(), dproduct, 3, 2, 3, 4, true);
    cudaDeviceSynchronize();
    cudaMemcpy(product, dproduct, 8*sizeof(float), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    for(int i = 0; i < 8; i++) {
        EXPECT_EQ(product[i], correct_ans[i]);
    }
    cudaFree(dproduct);
}

TEST(Main, MultiThreadedDotProductTransposeMatrix2) {
    int nWorkers = 1, nThreadsPerWorker = 1;
    dim3 nBlocks(nWorkers, 2, 1); //deals with the rows in the first matrix
    dim3 nThreads(nThreadsPerWorker, 2, 1); //deals with the rows in the second matrix
    float arr1[6] = {1,2,3,4,5,6}; // 2 x 3 matrix
    float arr2[12] = {-1,-5,-9,-2,-6,-10,-3,-7,-11,-4,-8,-12}; // 4 x 3 matrix
    float product[8];
    float correct_ans[8] = {-38.0f, -44.0f, -50.0f, -56.0f, -83.0f, -98.0f, -113.0f, -128.0f};
    std::shared_ptr<float> darr1 = transferMatrixToDevice(arr1, 2, 3);
    std::shared_ptr<float> darr2 = transferMatrixToDevice(arr2, 3, 4);
    float *dproduct;
    cudaMalloc(&dproduct, 8*sizeof(float));
    dotProductTransposeSegmented<<<nBlocks, nThreads>>>(darr1.get(), darr2.get(), dproduct, 2, 3, 4, 3, true);
    cudaDeviceSynchronize();
    cudaMemcpy(product, dproduct, 8*sizeof(float), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    for(int i = 0; i < 8; i++) {
        EXPECT_EQ(product[i], correct_ans[i]);
    }
    nBlocks.y = 1;
    dotProductTransposeSegmented<<<nBlocks, nThreads>>>(darr1.get(), darr2.get(), dproduct, 2, 3, 4, 3, true);
    cudaDeviceSynchronize();
    cudaMemcpy(product, dproduct, 8*sizeof(float), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    for(int i = 0; i < 8; i++) {
        EXPECT_EQ(product[i], correct_ans[i]);
    }
    nThreads.y = 1;
    dotProductTransposeSegmented<<<nBlocks, nThreads>>>(darr1.get(), darr2.get(), dproduct, 2, 3, 4, 3, true);
    cudaDeviceSynchronize();
    cudaMemcpy(product, dproduct, 8*sizeof(float), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    for(int i = 0; i < 8; i++) {
        EXPECT_EQ(product[i], correct_ans[i]);
    }
    cudaFree(dproduct);
}