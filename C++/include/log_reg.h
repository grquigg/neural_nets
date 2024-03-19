#ifndef LOG_REG_H
#define LOG_REG_H

#include <vector>

struct LogisticRegression {
    int nFeatures;
    int nClasses;
    float * weights;
    float * bias;
    float lambda;
    float * gradients;
};

LogisticRegression * copyModelToGPU(LogisticRegression *model, int nWorkers, int nThreadsPerWorker);

void train(LogisticRegression *model, float* train_input, std::vector<std::vector<int>>& train_labels, float* test_input, std::vector<std::vector<int>>& test_labels, 
int nEpochs, int batch_size, int total_size, int test_size, float learning_rate, int nWorkers, int nThreadsPerWorker);

__global__ void ringReduce(LogisticRegression * model, const int total_steps, const int step_size, const int chunk_size);

__global__ void backward_pass(LogisticRegression* model, int batch_size, float learning_rate);

__global__ void forward_pass(LogisticRegression* model, float* inputs, float* outputs, float* product, int size, int nClasses);

__global__ void predict(LogisticRegression* model, float* inputs, float* product, int size);

#endif