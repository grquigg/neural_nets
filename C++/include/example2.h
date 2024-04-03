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
        float expected[18] = {
            0.67699581f, 0.75384f, 0.58816868f, 0.7056604f,
            0.63471538f, 0.69291866f, 0.54391158f, 0.64659375f,
            0.87519467f, 0.8929618f, 0.81480443f,
            0.86020094f, 0.8833645f, 0.79790765f,
            0.48506981f, 0.51493f,
            0.4841319f, 0.51586807f
        };
        std::vector<dim3> *forward_pass_blockDim;
        std::vector<dim3> *forward_pass_gridDim;
        float ** weights;
        float ** biases;
        float *input;
        float *correctForward; 
        float *d_inputs;

        void SetUp() override;

        void TearDown() override;
};


#endif