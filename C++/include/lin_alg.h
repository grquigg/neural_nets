#ifndef LIN_ALG_H
#define LIN_ALG_H

#include <vector>
#include "../include/models.h"

//////////DEVICES/GENERAL LINEAR ALGEBRA FUNCTIONS////////

__device__ void reLU(float* mat, int height, int width);

__device__ void softmax(float* product, int product_height, int product_width);

__device__ void matrixSubtract(float * matrix1, float *matrix2, int m1_h, int m1_w, int m2_h, int m2_w, float * outVec);

__device__ void matrixAdd(float * matrix1, float * matrix2, int m1_h, int m1_w);

__device__ void matrixMultiplyByScalar(float* matrix, int m1_h, int m1_w, float scalar);

__device__ void sigmoid(float* inputs, int size);
__device__ void sigmoidD(float* activations, int height, int width, float * delta);

__device__ void dotProduct(float* inputs, float* weights, float * product, int vector_h, int vector_w, int weight_h, int weight_w);

__device__ void dotProduct(float* inputs, float* weights, float * product, int vector_h, int vector_w, int weight_h, int weight_w, float* bias);
//we can't unneccessarily waste memory on the GPU so I have to get creative and take a second attempt at writing the modified dotProduct function
__device__ float* transposeMatrix(float * matrix, int matrix_height, int matrix_width);

__global__ void setTranspose(NeuralNetwork* model);

__device__ void dotProductTranspose(float* inputs, float* weights, float * product, int vector_h, int vector_w, int weight_h, int weight_w);

//////////GLOBALS////////

__global__ void predict(LogisticRegression* model, float* inputs, float* product, int size);

__global__ void predict(float * inputs, float* weights, float * product, int size, int n_features, int n_classes);

__global__ void forward_pass(float* inputs, float* weights, float* outputs, float* product, float* gradients, int size, int n_features, int n_classes);

__global__ void forward_pass(LogisticRegression* model, float* inputs, float* outputs, float* product, int size, int nClasses);

__global__ void backward_pass(float* weights, float * gradients, int batch_size, float learning_rate, int n_features, int n_classes);

__global__ void backward_pass(LogisticRegression* model, int batch_size, float learning_rate);
/*
The only issue that ringReduce might have in our current implementation is that we don't know how many steps we need to actually take
However, we can leverage the fact that each thread has awareness of the global dimensions, so that's how many partitions we need to use
*/
__global__ void ringReduce(float * gradients, const int total_steps, const int step_size, const int chunk_size);

__global__ void ringReduce(LogisticRegression * model, const int total_steps, const int step_size, const int chunk_size);

__global__ void backprop(NeuralNetwork* model, float* inputs, float* outputs, float* activations, float* deltas, int* offsets, int size, int nClasses);

__global__ void predict(NeuralNetwork* model, float* inputs, float* activations,  int* offsets, int size);

__global__ void ringReduce(NeuralNetwork* model, const int total_steps);

__global__ void backward_pass(NeuralNetwork* model, int batch_size, float learning_rate);

__global__ void auditGradients(NeuralNetwork* model);

__global__ void auditDeltas(NeuralNetwork* model, float* deltas, int* offsets, int batches, int batch_size);
#endif