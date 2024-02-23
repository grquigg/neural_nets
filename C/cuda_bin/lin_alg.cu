#include "../include/lin_alg_gpu.h"
#include <stdbool.h>
#include <stdlib.h>
#include <stdio.h>
#include <cuda_runtime.h>

__device__
void dotProduct(int weight_h, int weight_w, int vector_h, int vector_w, float * weights, float * vectors, float * product, int * count) {
    //if we have a matrix of H*W, then vector_h == weight_w
    if(weight_w != vector_h) {
        return;
    }
    int index = blockIdx.x*blockDim.x + threadIdx.x;
    //initialize the matrix
    for(int i = 0; i < weight_h; i++) {
        for(int j = 0; j < vector_w; j++) {
            (*count)++;
            product[i*vector_w+j] = 0.0f;
            //printf("%d %d %f\n", i, j, weights[i][j]);
            for(int k = 0; k < weight_w; k++) {
                product[i*vector_w+j] += weights[i*weight_w+k] * vectors[k*vector_w+j];
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

__device__
void matrixSubtract(float * matrix1, float *matrix2, int m1_h, int m1_w, int m2_h, int m2_w) {
    if (m1_h == m2_h && m1_w == m2_w) {
        for (int i = 0; i < m1_h; i++) {
            for (int j = 0; j < m1_w; j++) {
                matrix1[(i*m1_w)+j]-=matrix2[(i*m1_w)+j];
            }
        }
    }
}

void matrixAdd(float** matrix1, float** matrix2, float m1_h, float m1_w, float m2_h, float m2_w) {
    if (m1_h == m2_h && m1_w == m2_w) {
        for (int i = 0; i < m1_h; i++) {
            for (int j = 0; j < m1_w; j++) {
                matrix1[i][j] += matrix2[i][j];
            }
        }
    }
}

void multiplyMatrixByScalar(float** matrix, int matrix_height, int matrix_width, float scalar) {
    for (int i = 0; i < matrix_height; i++) {
        for (int j = 0; j < matrix_width; j++) {
            matrix[i][j] *= scalar;
        }
    }
}


void matrixMultiply(float **mat1, float **mat2, float height, float width) {
    for(int i = 0; i < height; i++) {
        for(int j = 0; j < width; j++) {
            mat1[i][j] *= mat2[i][j];
        }
    }
}

float** transposeMatrix(float ** matrix, int matrix_height, int matrix_width) {
    float **transpose;
    transpose = (float**) malloc(sizeof(float**) * matrix_width);
    for(int i = 0; i < matrix_width; i++) {
        transpose[i] = (float*)malloc(sizeof(float) * matrix_height);
        for(int j = 0; j < matrix_height; j++) {
            transpose[i][j] = matrix[j][i];
        }
    }
    return transpose;
}

__device__
void softmax(float* product, int product_height, int product_width) {
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

__global__
void forward_pass(int BATCH_SIZE, float* inputs, float* weights, float* outputs, float*product, int size, int n_features, int n_classes, int * counter) {
    //the dot product has been modified to account for the fact that we're passing 1D arrays
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    //BATCH_SIZE*n_classes length vector
    dotProduct(BATCH_SIZE, n_features, n_features, n_classes, weights, inputs+(i*BATCH_SIZE*n_features), product+(i*BATCH_SIZE*n_classes), counter);
    printf("End %d %d\n", i, *counter);
    // softmax(product+(i*BATCH_SIZE*n_classes), BATCH_SIZE, n_classes);
}