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
    int BATCH_SIZE = 4000;
    int nEpochs = 100;
    float learning_rate = 0.02;
    int nWorkers = 16;
    int nThreadsPerWorker = 10;

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
    //define our model on the client side
    LogisticRegression* model = (LogisticRegression *)malloc(sizeof(LogisticRegression));
    model->nFeatures = nFeatures;
    model->nClasses = numClasses;
    model->weights = initializeFlatRandomArray(nFeatures, numClasses);
    model->gradients = (float*)malloc(nThreadsPerWorker*nWorkers*nFeatures*numClasses*sizeof(float));

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
    return 0;
}