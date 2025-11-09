#include <gtest/gtest.h>
#include "../include/lin_alg.h"
#include "../include/utils.h"
#include "../include/models.h"

TEST(NeuralNetwork, TestActivationsForExOne) {
    int batch_size = 1;
    int * layers = new int[3]{1,2, 1};
    int nLayers = 2;
    float input[2] = {0.13000f, 0.42f};
    float **weights = new float*[2];
    weights[0] = new float[2]{0.1f, 0.2f};
    weights[1] = new float[2]{0.5f, 0.6f};
    float **biases = new float*[2];
    biases[0] = new float[2]{0.4f, 0.3f};
    biases[1] = new float[1]{0.7f};
    NeuralNetwork model(nLayers, layers, weights, biases, 1.0f);
    float correctOutput[3] = {0.601807f,0.58078581f, 0.79403f};
    model.setupGPU(1, batch_size);
    model.final_activation = sigmoidHost;
    std::shared_ptr<float> d_input = transferMatrixToDevice(input, 1, 1);
    std::shared_ptr<float> activations = model.forward_pass(d_input, 1, 1, 1, 1);
    for(int i = 0; i < 3; i++) {
        std::cout << i << std::endl;
        EXPECT_FLOAT_EQ(activations.get()[i], correctOutput[i]);
    }
}
