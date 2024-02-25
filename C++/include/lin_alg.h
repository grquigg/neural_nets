#ifndef LIN_ALG_H
#define LIN_ALG_H

#include <vector>

__device__ void matrixSubtract(float * matrix1, float *matrix2, int m1_h, int m1_w, int m2_h, int m2_w, float scalar);

__device__ void softmax(float* product, int product_height, int product_width);

__device__ void dotProduct(float* inputs, float* weights, float * product, int vector_h, int vector_w, int weight_h, int weight_w);

//we can't unneccessarily waste memory on the GPU so I have to get creative and take a second attempt at writing the modified dotProduct function
__device__ void dotProductTranspose(float* inputs, float* weights, float * product, int vector_h, int vector_w, int weight_h, int weight_w);

__global__ void forward_pass(float* inputs, float* weights, float* outputs, float* product, float* gradients, int size, int n_features, int n_classes);
#endif