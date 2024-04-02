#include <gtest/gtest.h>
#include "models.h"
#include "example2.h"
/*
This class is to simplify the process of testing the Neural Network on the second example
that I was given as part of the CSCI589 class. This will ideally allow us to test attributes
of the same network without having to explicitly instantiate it every single time. 
*/

void Example2Suite::SetUp() {
    weights = new float*[nLayers];
    correctForward = new float*[nLayers];
    correctForward[0] = new float[8]{0.74f, 1.1192f, 0.3564f, 0.8744f, 0.55250f, 0.81380f, 0.17610f, 0.60410f};
    input = new float[4]{0.32f, 0.68f, 0.83f, 0.02f};
    weights[0] = new float[8]{0.15f, 0.1f, 0.19f, 0.35f, 0.4f, 0.54f, 0.42f, 0.68f};
    weights[1] = new float[12]{0.67f, 0.42f, 0.56f, 0.14f, 0.2f, 0.8f, 0.96f, 0.32f, 0.69f, 0.87f, 0.89f, 0.09f};
    weights[2] = new float[6]{0.87f, 0.1f, 0.42f, 0.95f, 0.53f, 0.69f};
    biases = new float*[3];
    biases[0] = new float[4]{0.42f, 0.72f, 0.01f, 0.3f};
    biases[1] = new float[3]{0.21f, 0.87f, 0.03f};
    biases[2] = new float[2]{0.04f, 0.17f};
    model = new NeuralNetwork(nLayers, layers, weights, biases, 1.0f);
}

void Example2Suite::TearDown() {
    delete model;
}