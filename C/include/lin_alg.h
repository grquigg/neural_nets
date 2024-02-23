#ifndef LIN_ALG_H
#define LIN_ALG_H

float** transposeMatrix(float ** matrix, int matrix_height, int matrix_width);

void multiplyMatrixByScalar(float** matrix, int matrix_height, int matrix_width, float scalar);

void matrixAdd(float** matrix1, float** matrix2, float m1_h, float m1_w, float m2_h, float m2_w);

void matrixSubtract(float ** matrix1, float **matrix2, float m1_h, float m1_w, float m2_h, float m2_w, float scalar);

void matrixMultiply(float **mat1, float **mat2, float height, float width);

void dotProduct(int weight_h, int weight_w, int vector_h, int vector_w, float ** weights, float ** vectors, float ** product, int * counter);

#endif