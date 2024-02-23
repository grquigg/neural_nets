#ifndef LIN_ALG_GPU_H
#define LIN_ALG_GPU_H

__device__
void dotProduct(int weight_h, int weight_w, int vector_h, int vector_w, float * weights, float * vectors, float * product);

__device__
void softmax(float* product, int product_height, int product_width);

__device__
void matrixSubtract(float * matrix1, float *matrix2, float m1_h, float m1_w, float m2_h, float m2_w);

__global__
void forward_pass(int BATCH_SIZE, float* inputs, float* weights, float* outputs, float* product, int size, int n_features, int n_classes, int * counter);


#endif