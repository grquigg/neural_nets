#ifndef LIN_ALG_H
#define LIN_ALG_H

#include <vector>

/*
The only issue that ringReduce might have in our current implementation is that we don't know how many steps we need to actually take
However, we can leverage the fact that each thread has awareness of the global dimensions, so that's how many partitions we need to use
*/
__global__ void ringReduce(float * gradients, const int total_steps, const int step_size, const int chunk_size);

__device__ void matrixSubtract(float * matrix1, float *matrix2, int m1_h, int m1_w, int m2_h, int m2_w, float scalar);

__device__ void matrixAdd(float * matrix1, float * matrix2, int m1_h, int m1_w);

__device__ void matrixMultiplyByScalar(float* matrix, int m1_h, int m1_w, float scalar);

__device__ void softmax(float* product, int product_height, int product_width);

__device__ void dotProduct(float* inputs, float* weights, float * product, int vector_h, int vector_w, int weight_h, int weight_w);

//we can't unneccessarily waste memory on the GPU so I have to get creative and take a second attempt at writing the modified dotProduct function
__device__ void dotProductTranspose(float* inputs, float* weights, float * product, int vector_h, int vector_w, int weight_h, int weight_w);

__global__ void forward_pass(float* inputs, float* weights, float* outputs, float* product, float* gradients, int size, int n_features, int n_classes);

__global__ void backward_pass(float* weights, float * gradients, int batch_size, float learning_rate, int n_features, int n_classes);
#endif