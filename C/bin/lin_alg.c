#include "../include/lin_alg.h"
#include <stdbool.h>
#include <stdlib.h>
#include <stdio.h>

void matrixSubtract(float ** matrix1, float **matrix2, float m1_h, float m1_w, float m2_h, float m2_w, float scalar) {
    if (m1_h == m2_h && m1_w == m2_w) {
        for (int i = 0; i < m1_h; i++) {
            for (int j = 0; j < m1_w; j++) {
                matrix1[i][j]-=matrix2[i][j];
                matrix1[i][j] *= scalar;
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

void dotProduct(int weight_h, int weight_w, int vector_h, int vector_w, float ** weights, float ** vectors, float ** product) {
    //if we have a matrix of H*W, then vector_h == weight_w
    if(weight_w != vector_h) {
        printf("INVALID VALUES FOR MATRIX AND VECTOR\n");
        return;
    }
    //initialize the matrix
    for(int i = 0; i < weight_h; i++) {
        product[i] = (float*)malloc(vector_w * sizeof(float));
        //printf("Success %d\n", i);
        for(int j = 0; j < vector_w; j++) {
            product[i][j] = 0;
            //printf("New entry\n");
            //printf("%d %d %f\n", i, j, weights[i][j]);
            for(int k = 0; k < weight_w; k++) {
                product[i][j] += weights[i][k] * vectors[k][j];
                if (product[i][j] > 10000) {
                    printf("Problem children from normal dot product: %.15f %.15f\n", weights[i][k], vectors[k][j]);
                    printf("Product at %d %d: %f\n", i, j, product[i][j]);
                }
                //printf("Temp product: %f\n", product[i][j]);
            }
            //printf("%f\n", product[i][j]);
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
