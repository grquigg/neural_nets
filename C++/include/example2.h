#ifndef EXAMPLE_2
#define EXAMPLE_2
#include <gtest/gtest.h>
#include "models.h"
/*
This class is to simplify the process of testing the Neural Network on the second example
that I was given as part of the CSCI589 class. This will ideally allow us to test attributes
of the same network without having to explicitly instantiate it every single time. 
*/
class Example2Suite: public ::testing::Test {
    protected:
        NeuralNetwork *model;
        int nLayers = 3;
        int n_inputs = 2;
        int layers[4] = {2, 4, 3, 2};
        float ** weights;
        float ** biases;
        float *input;
        float **correctForward; 
        std::shared_ptr<float> d_inputs;

        void SetUp() override;

        void TearDown() override;
};


#endif