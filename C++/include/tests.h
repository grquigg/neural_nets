#ifndef TESTS_H
#define TESTS_H

#include <vector>
#include <cassert>

void testDotProducts();

__global__ void testfunc();

void dotProduct(float* inputs, float* weights, float * product, int vector_h, int vector_w, int weight_h, int weight_w);

void dotProductTranspose(float* inputs, float* weights, float * product, int vector_h, int vector_w, int weight_h, int weight_w);

void testDotProduct(float* arr1, float* arr2, float* arr1_T, float* arr2_T, float*product1, float* product2, float* product3);
#endif

