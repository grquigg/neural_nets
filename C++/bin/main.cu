#include <iostream>
#include <cassert>
#include <string>
#include <vector>
// #include "utils.h"
#include "../include/utils.h"
#include "../include/lin_alg.h"
#include "../include/models.h"
#include <chrono> 

int main(int argc, char** argv) {
    ////HYPERPARAMS////
    std::string train_data_path = argv[1];
    std::string train_label_path = argv[2];
    std::string test_data_path = argv[3];
    std::string test_label_path = argv[4];
    // std::string nWorkers_arg = argv[5];
    // std::string nThreads_arg = argv[6];
    int numClasses = 10;
    int BATCH_SIZE = std::stoi(argv[7]);
    int nEpochs = 10;
    float learning_rate = 0.01;
    int nWorkers = std::stoi(argv[5]);
    int nThreadsPerWorker = std::stoi(argv[6]);

    //read in input file
    std::vector<std::vector<int>> inputs = readDataFromUByteFile(train_data_path);
    std::vector<std::vector<int>> test_inputs = readDataFromUByteFile(test_data_path);
    int size = inputs.size();
    int nFeatures = inputs[0].size();
    int test_size = test_inputs.size();
    
    std::vector<float> input(size*nFeatures, 0.0f);
    std::vector<float> test_input(test_size*nFeatures, 0.0f);


    for(int i = 0; i < size; i++) {
        for(int j = 0; j < nFeatures; j++) {
            input[i*nFeatures + j] = (float) inputs[i][j] / 255.0;
        }
    }
    // input[0] = 0.32;
    // input[1] = 0.68;
    // input[2] = 0.83;
    // input[3] = 0.02;
    for(int i = 0; i < test_size; i++) {
        for(int j = 0; j < nFeatures; j++) {
            test_input[i*nFeatures + j] = (float) test_inputs[i][j] / 255.0;
        }
    }
    //read in output file
    std::vector<std::vector<int>> outputs = readDataFromUByteFile(train_label_path);
    std::vector<std::vector<int>> test_outputs = readDataFromUByteFile(test_label_path);
    assert(test_outputs.size() == 10000);
    ////MODEL/GPU LOGIC////
    //define our model on the host side
    //LOGISTIC REGRESSION
    // LogisticRegression* model = (LogisticRegression *)malloc(sizeof(LogisticRegression));
    // model->nFeatures = nFeatures;
    // model->nClasses = numClasses;
    // model->weights = initializeFlatRandomArray(nFeatures, numClasses);
    // model->gradients = (float*)malloc(nThreadsPerWorker*nWorkers*nFeatures*numClasses*sizeof(float));

    //NEURAL NETWORK
    NeuralNetwork* model = new NeuralNetwork;
    model->nClasses = numClasses;
    model->nLayers = 2;
    model->lambda = 1.0;
    model->layer_size = (int*)malloc((model->nLayers+1)*sizeof(int));
    model->layer_size[0] = 784;
    model->layer_size[1] = 800;
    model->layer_size[2] = 10;
    model->weights = (float**)malloc((model->nLayers)*sizeof(float*));
    model->biases = (float**)malloc((model->nLayers)*sizeof(float*));
    model->gradients = (float**)malloc((model->nLayers)*sizeof(float*));
    model->grad_biases =(float**)malloc((model->nLayers)*sizeof(float*));
    // model->weights[0] = (float*)malloc(8*sizeof(float));
    // model->weights[0][0] = 0.15;
    // model->weights[0][1] = 0.1;
    // model->weights[0][2] = 0.19;
    // model->weights[0][3] = 0.35;
    // model->weights[0][4] = 0.40;
    // model->weights[0][5] = 0.54;
    // model->weights[0][6] = 0.42;
    // model->weights[0][7] = 0.68;
    // model->weights[1] = (float*)malloc(12*sizeof(float));
    // model->weights[1][0] = 0.67;
    // model->weights[1][1] = 0.42;
    // model->weights[1][2] = 0.56;
    // model->weights[1][3] = 0.14;
    // model->weights[1][4] = 0.2;
    // model->weights[1][5] = 0.8;
    // model->weights[1][6] = 0.96;
    // model->weights[1][7] = 0.32;
    // model->weights[1][8] = 0.69;
    // model->weights[1][9] = 0.87;
    // model->weights[1][10] = 0.89;
    // model->weights[1][11] = 0.09;
    // model->weights[2] = (float*)malloc(6*sizeof(float));
    // model->weights[2][0] = 0.87;
    // model->weights[2][1] = 0.10;
    // model->weights[2][2] = 0.42;
    // model->weights[2][3] = 0.95;
    // model->weights[2][4] = 0.53;
    // model->weights[2][5] = 0.69;
    // model->biases[0] = (float*)malloc(4*sizeof(float));
    // model->biases[0][0] = 0.42;
    // model->biases[0][1] = 0.72;
    // model->biases[0][2] = 0.01;
    // model->biases[0][3] = 0.3;
    // model->biases[1] = (float*)malloc(3*sizeof(float));
    // model->biases[1][0] = 0.21;
    // model->biases[1][1] = 0.87;
    // model->biases[1][2] = 0.03;
    // model->biases[2] = (float*)malloc(2*sizeof(float));
    // model->biases[2][0] = 0.04;
    // model->biases[2][1] = 0.17;
    for(int i = 1; i < model->nLayers+1; i++) {
        model->weights[i-1] = initializeFlatRandomArray(model->layer_size[i-1], model->layer_size[i]);
        model->biases[i-1] = initializeFlatRandomArray(1, model->layer_size[i]);
        model->grad_biases[i-1] = (float*)malloc(nThreadsPerWorker*nWorkers*model->layer_size[i]);
        model->gradients[i-1] =(float*)malloc(nThreadsPerWorker*nWorkers*model->layer_size[i-1]*model->layer_size[i]);
    }
    //pass training function
    //the train function takes all of the HOST variables and passes them to the GPU before running the training algorithm
    /*
    The order of the arguments that should be passed into the train function are as follows:
    1. Reference to the HOST model struct
    2. Reference to the train input array
    3. Reference to the train labels array
    4. Reference to the test input array
    5. Reference to the test labels array
    ////HYPERPARAMS
    6. Number of epochs
    7. Batch size
    8. Full size of the entire train set
    9. Full size of the entire test set
    10. learning rate
    11. number of desired workers
    12. number of threads per worker
    */
    train(model, input.data(), outputs, test_input.data(), test_outputs, nEpochs, BATCH_SIZE, size, test_size, learning_rate, nWorkers, nThreadsPerWorker);
    free(model);
    return 0;
}