#ifndef LIN_ALG_H
#define LIN_ALG_H

#include <vector>

__device__ void dotProduct(float* inputs, float* weights, float * product, int vector_h, int vector_w, int weight_h, int weight_w);

__global__ void forward_pass(float* inputs, float* weights, float* outputs, float* product, int size, int n_features, int n_classes);
#endif