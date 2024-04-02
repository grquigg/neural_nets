#ifndef MODELS_H
#define MODELS_H
#include <memory>
std::shared_ptr<float> transferMatrixToDevice(float *matrix, int height, int width);

std::shared_ptr<int> transferMatrixToDevice(int * matrix, int height, int width);

void free2DArrayFromDevice(float ** array, int * array_size);

struct CudaDeallocator {
    void operator()(float* ptr) {
        cudaFree(ptr);
    }

    void operator()(int* ptr) {
        cudaFree(ptr);
    }
};

struct LogisticRegression {
    int nFeatures;
    int nClasses;
    float * weights;
    float * bias;
    float lambda;
    float * gradients;
};

class NeuralNetwork {
    public:
        int nClasses;
        int nLayers;
        int * layer_size;
        float ** weights;
        float ** d_weights;
        float ** biases;
        float ** d_biases;
        float lambda;
        float ** gradients;
        float ** grad_biases;
        bool on_device = false;
        float * activations;
        float ** deltas;
        int * offsets;
        std::vector<dim3> forward_pass_specs;
        std::vector<dim3> backward_pass_specs;
        NeuralNetwork();
        NeuralNetwork(int nLayers, int * layer_size);
        NeuralNetwork(int nLayers, int * layer_size, float** weights, float ** biases, float lambda);
        ~NeuralNetwork();

        void train();

        void setupDeltas(int batch_size);

        std::shared_ptr<float> forward_pass(std::shared_ptr<float> d_input, int total_size, int batch_size, int nWorkers, int nThreadsPerWorkers) ;

        void setupGPU(int nWorkers, int batch_size);

        void backprop(int batch_size, std::shared_ptr<float> inputs, std::shared_ptr<float> outputs);
};

LogisticRegression * copyModelToGPU(LogisticRegression *model, int nWorkers, int nThreadsPerWorker);

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
int nEpochs, int batch_size, int total_size, int test_size, float learning_rate, int nWorkers, int nThreadsPerWorker, bool useMultiThreaded);

NeuralNetwork* buildModel(int nLayers, int * layer_size, float** weights, float **biases, float lambda, int nThreadsPerWorker, int nWorkers);

#endif