#include <vector>
#include <iostream>
#include "../include/lin_alg.h"


__global__ void forward_pass(float* inputs, float* weights, float* outputs, float* product, int size, int n_features, int n_classes) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    //BATCH_SIZE*n_classes length vector
    dotProduct(inputs+(i*size*n_features), weights, product+(i*size*n_classes), size, n_features, n_features, n_classes);
}

__device__ void dotProduct(float* inputs, float* weights, float * product, int vector_h, int vector_w, int weight_h, int weight_w) {
    int index = blockIdx.x*blockDim.x + threadIdx.x;
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