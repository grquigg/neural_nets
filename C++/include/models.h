#ifndef MODELS_H
#define MODELS_H
struct LogisticRegression {
    int nFeatures;
    int nClasses;
    float * weights;
    float * bias;
    float lambda;
    float * gradients;
};

struct NeuralNetwork {
    int nClasses;
    int nLayers;
    int * layer_size;
    float ** weights;
    float ** biases;
    float lambda;
    float ** gradients;
    float ** grad_biases;
};

LogisticRegression * copyModelToGPU(LogisticRegression *model, int nWorkers, int nThreadsPerWorker);

NeuralNetwork * copyModelToGPU(NeuralNetwork *model, int nWorkers, int nThreadsPerWorker);
void copyDataToGPU(float* train_input, std::vector<std::vector<int>>& train_labels, float* test_input, std::vector<std::vector<int>>& test_labels, int total_size, int test_size, int nClasses);
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
9. learning rate
10. number of desired workers
11. number of threads per worker
*/
void train(LogisticRegression *model, float* train_input, std::vector<std::vector<int>>& train_labels, float* test_input, std::vector<std::vector<int>>& test_labels, 
int nEpochs, int batch_size, int total_size, int test_size, float learning_rate, int nWorkers, int nThreadsPerWorker);

void train(NeuralNetwork *model, float* train_input, std::vector<std::vector<int>>& train_labels, float* test_input, std::vector<std::vector<int>>& test_labels, 
int nEpochs, int batch_size, int total_size, int test_size, float learning_rate, int nWorkers, int nThreadsPerWorker);

NeuralNetwork* buildModel(int nLayers, int * layer_size, float** weights, float **biases, float lambda, int nThreadsPerWorker, int nWorkers);

#endif