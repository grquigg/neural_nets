#include <vector>
#include <iostream>
#include "../include/lin_alg.h"


__global__ void forward_pass(float* inputs, float* weights, float* outputs, float* product, float* gradients, int size, int n_features, int n_classes) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    //BATCH_SIZE*n_classes length vector
    dotProduct(inputs+(i*size*n_features), weights, product+(i*size*n_classes), size, n_features, n_features, n_classes);
    softmax(product+(i*size*n_classes), size, n_classes);
    matrixSubtract(product+(i*size*n_classes), outputs+(i*size*n_classes), size, n_classes, size, n_classes, -1);
    dotProductTranspose(inputs+(i*size*n_features), product+(i*size*n_classes), gradients+(i*n_features*n_classes), size, n_features, size, n_classes);
}

__device__ void softmax(float* product, int product_height, int product_width) {
    float total = 0.0;
    float logSumTotal = 0.0;
    for (int i = 0; i < product_height; i++) {
        total = 0.0;
        for (int j = 0; j < product_width; j++) {
            total += exp(product[i*product_width+j]);
        }
        logSumTotal = logf(total);
        float prob_sums = 0.0;
        for (int j = 0; j < product_width; j++) {
            product[i*product_width+j] = exp(product[i*product_width+j] - logSumTotal);
            prob_sums += product[i*product_width+j];
        }
        
    }
}
__device__ void dotProduct(float* inputs, float* weights, float * product, int vector_h, int vector_w, int weight_h, int weight_w) {
    //initialize the matrix
    //dot product is ALWAYS computed as the rows of the first matrix by the columns of the second matrix
    for(int i = 0; i < vector_h; i++) { //for every row in the first matrix
        for(int j = 0; j < weight_w; j++) { //for every column in the second matrix
            product[i*weight_w+j] = 0.0f;
            //printf("%d %d %f\n", i, j, weights[i][j]);
            for(int k = 0; k < vector_w; k++) { //we compute the kth entry in row i of the INPUTS times the kth entry in column j of the WEIGHTS
                product[i*weight_w+j] += inputs[i*vector_w+k] * weights[k*weight_w+j];
                if (product[i*vector_w+j] > 10000) {
                    // printf("Problem children from normal dot product: %.15f %.15f\n", weights[i][k], vectors[k][j]);
                    // printf("Product at %d %d: %f\n", i, j, product[i][j]);
                }
                //printf("Temp product: %f\n", product[i][j]);
            }
            //printf("%f\n", product[i][j]);
        }
    }
}

__device__ void dotProductTranspose(float* inputs, float* weights, float * product, int vector_h, int vector_w, int weight_h, int weight_w) {
    //remember that we want the resulting matrix to be of shape [vector_h, weight_w]
    for(int i = 0; i < vector_w; i++) {
        for(int j = 0; j < weight_w; j++) {
            product[i*weight_w+j] = 0.0;
            for(int k = 0; k < vector_h; k++) {
                product[i*weight_w+j] += inputs[k*vector_w+i] * weights[k*weight_w+j];
            }
        }
    }
}

__device__ void matrixSubtract(float * matrix1, float *matrix2, int m1_h, int m1_w, int m2_h, int m2_w, float scalar) {
    if (m1_h == m2_h && m1_w == m2_w) {
        for (int i = 0; i < m1_h; i++) {
            for (int j = 0; j < m1_w; j++) {
                matrix1[(i*m1_w)+j]-=matrix2[(i*m1_w)+j];
                matrix1[(i*m1_w)+j]*=-1;
            }
        }
    }
}