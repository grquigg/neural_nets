#include <iostream>
#include <cassert>
#include <string>
#include <vector>
// #include "utils.h"
#include "../include/utils.h"
#include "../include/lin_alg.h"
#include "../include/models.h"
#include <chrono>
#include <memory>

int main(int argc, char** argv) {
    ////HYPERPARAMS////
    std::string root = argv[1];
    std::string train_data_path = root + "/train-images.idx3-ubyte";
    std::string train_label_path = root + "/train-labels.idx1-ubyte";
    std::string test_data_path = root + "/t10k-images.idx3-ubyte";
    std::string test_label_path =  root + "/t10k-labels.idx1-ubyte";
    // std::string nWorkers_arg = argv[5];
    // std::string nThreads_arg = argv[6];
    int numClasses = 10;
    int BATCH_SIZE = std::stoi(argv[2]);
    int nEpochs = 1;
    float learning_rate = 0.025;

    /*
    nWorkers*nThreadsPerWorker is the number of general threads we have working on each batch of memory.
    It also corresponds with how many copies of the gradients (for both weights and biases) and activations we make for each batch, and should 
    ideally remain the same for the ring reduce method. 
    Ring reduce is the ONLY global function that directly relies on this number. 
    The other global functions rely on it in part, but they can technically be abstracted from one another. 
    */
    int nWorkers = std::stoi(argv[3]);
    int nThreadsPerWorker = std::stoi(argv[4]);

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

    //NEURAL NETWORK
    int * layer_size = new int[3]{784,16,10};
    NeuralNetwork model(2, layer_size);


    model.setupGPU(nThreadsPerWorker*nWorkers, BATCH_SIZE);
    std::shared_ptr<float> d_input = transferMatrixToDevice(input.data(), BATCH_SIZE, model.layer_size[0]);
    model.forward_pass(d_input, 2, BATCH_SIZE, nWorkers, nThreadsPerWorker);
    
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
    // train(model, input.data(), outputs, test_input.data(), test_outputs, nEpochs, BATCH_SIZE, size, test_size, learning_rate, nWorkers, nThreadsPerWorker, true);
    // free(model);
    return 0;
}