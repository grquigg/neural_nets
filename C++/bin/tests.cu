#include <iostream>
#include <cassert>
#include <string>
#include <vector>
#include "../include/tests.h"
#include "../include/utils.h"
#include "../include/lin_alg.h"

float* transferMatrixToDevice(float *matrix, int height, int width) {
    float* deviceMatrix;
    cudaMalloc(&deviceMatrix, height*width*sizeof(float));
    for(int i = 0; i < height; i++) {
        cudaMemcpy(deviceMatrix+(i*width), matrix+(i*width), sizeof(float)*width, cudaMemcpyHostToDevice);
    }
    return deviceMatrix;
}

// void dotProduct(float* inputs, float* weights, float * product, int vector_h, int vector_w, int weight_h, int weight_w) {
//     //initialize the matrix
//     //dot product is ALWAYS computed as the rows of the first matrix by the columns of the second matrix
//     if (vector_w != weight_h) {
//         printf("invalid values\n");
//         return;
//     }
//     for(int i = 0; i < vector_h; i++) { //for every row in the first matrix
//         for(int j = 0; j < weight_w; j++) { //for every column in the second matrix
//             product[i*weight_w+j] = 0.0;
//             for(int k = 0; k < vector_w; k++) { //we compute the kth entry in row i of the INPUTS times the kth entry in column j of the WEIGHTS
//                 product[i*weight_w+j] += inputs[i*vector_w+k] * weights[k*weight_w+j];
//                 // printf("This %d %d %f %f\n", i, j, inputs[i*vector_w+k], weights[k*weight_w+j]);
//             }
//             // printf("%f\n", product[i*weight_w+j]);
//         }
//     }
// }

// void dotProductTranspose(float* inputs, float* weights, float * product, int vector_h, int vector_w, int weight_h, int weight_w) {
//     //remember that we want the resulting matrix to be of shape [vector_h, weight_w]
//     if(vector_h == weight_h) {
//     for(int i = 0; i < vector_w; i++) {
//         for(int j = 0; j < weight_w; j++) {
//             product[i*weight_w+j] = 0.0;
//             for(int k = 0; k < vector_h; k++) {
//                 product[i*weight_w+j] += inputs[k*vector_w+i] * weights[k*weight_w+j];
//             }
//         }
//     }
//     } else if(vector_w == weight_w) {
//         for(int i = 0; i < vector_h; i++) {
//             for(int j = 0; j < weight_h; j++) {
//                 product[i*weight_h+j] = 0.0;
//                 for(int k = 0; k < vector_w; k++) {
//                     product[i*weight_h+j] += inputs[i*vector_w+k] * weights[j*weight_w+k];
//                 }
//             }
//         }
//     } else {
//         printf("INVALID DIMS FOR DOT PRODUCT\n");
//     }
// }

// void testDotProduct(float* arr1, float* arr2, float* arr1_T, float* arr2_T, float*product1, float* product2, float* product3) {
//     dotProduct(arr1, arr2, product1, 2, 3, 3, 4);
// }

// void testDotProducts() {
//     printf("Test\n");
    // float arr1[6] = {1,2,3,4,5,6};
    // float arr1_T[6] = {1,4,2,5,3,6};
    // float arr2[12] = {-1,-2,-3,-4,-5,-6,-7,-8,-9,-10,-11,-12};
    // float arr2_T[12] = {-1,-5,-9,-2,-6,-10,-3,-7,-11,-4,-8,-12};
//     printf("Here\n");
//     float product1[8];
//     float product2[8];
//     float product3[8];
//     dotProduct(arr1, arr2, product1, 2, 3, 3, 4);
//     for(int i = 0; i < 8; i++) {
//         std::cout << product1[i] << std::endl;
//     }
//     dotProductTranspose(arr1_T, arr2, product2, 3, 2, 3, 4);
//     for(int i = 0; i < 8; i++) {
//         std::cout << product2[i] << std::endl;
//     }
//     dotProductTranspose(arr1, arr2_T, product3, 2, 3, 4, 3);
//     for(int i = 0; i < 8; i++) {
//         std::cout << product3[i] << std::endl;
//     }
// }

void testSegmentedDotProduct(int nWorkers, int nThreadsPerWorker) {
    float arr1[6] = {1,2,3,4,5,6};
    float arr2[12] = {-1,-2,-3,-4,-5,-6,-7,-8,-9,-10,-11,-12};
    float product[8];
    float *darr1;
    float *darr2;
    float *dproduct;
    cudaMalloc(&darr1, 6*sizeof(float));
    cudaMalloc(&darr2, 12*sizeof(float));
    cudaMalloc(&dproduct, 8*sizeof(float));
    cudaMemcpy(darr1, arr1, 6*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(darr2, arr2, 12*sizeof(float), cudaMemcpyHostToDevice);
    dim3 nBlocks(nWorkers, 1, 1);
    dim3 nThreads(nThreadsPerWorker, 1, 1);
    dotProductSegmented<<<nBlocks, nThreads>>>(darr1, darr2, dproduct, 2, 3, 3, 4);
    cudaDeviceSynchronize();
    cudaMemcpy(product, dproduct, 8*sizeof(float), cudaMemcpyDeviceToHost);
    for(int i = 0; i < 2; i++) {
        for(int j = 0; j < 4; j++) {
            printf("%f\t", product[i*4+j]);
        }
        printf("\n");
    }
    nBlocks.y = 2;
    dotProductSegmented<<<nBlocks, nThreads>>>(darr1, darr2, dproduct, 2, 3, 3, 4);
    cudaDeviceSynchronize();
    cudaMemcpy(product, dproduct, 8*sizeof(float), cudaMemcpyDeviceToHost);
    for(int i = 0; i < 2; i++) {
        for(int j = 0; j < 4; j++) {
            printf("%f\t", product[i*4+j]);
        }
        printf("\n");
    }
    nBlocks.y = 4;
    nThreads.y = 2;
    dotProductSegmented<<<nBlocks, nThreads>>>(darr1, darr2, dproduct, 2, 3, 3, 4);
    cudaDeviceSynchronize();
    cudaMemcpy(product, dproduct, 8*sizeof(float), cudaMemcpyDeviceToHost);
    for(int i = 0; i < 2; i++) {
        for(int j = 0; j < 4; j++) {
            printf("%f\t", product[i*4+j]);
        }
        printf("\n");
    }
};

int main() {
    printf("Main\n");
    // testDotProducts();
    testSegmentedDotProduct(1, 1);
    // int test_size = 40;
    // int n_features = 20;
    // int out_classes = 5;
    // int nWorkers = 4;
    // int nThreadsPerWorker = 5;
    // float learning_rate = 0.01;
    // int BATCH_SIZE = 20;
    // int epochs = 100;
    // float * inputs = (float *)malloc(n_features*test_size*sizeof(float));
    // for(int i = 0; i < test_size; i++) {
    //     for(int j = 0; j < n_features; j++) {
    //         inputs[(i*n_features)+j] = 0;
    //     }
    //     inputs[(i*n_features) + (i % n_features)] = 1.0;
    // }
    // printMatrix(inputs, test_size, n_features);
    // float * outputs = (float*)malloc(test_size*out_classes*sizeof(float));
    // for(int i = 0; i < test_size; i++) {
    //     for(int j = 0; j < out_classes; j++) {
    //         outputs[i*out_classes+j] = 0.0;
    //     }
    //     outputs[(i*out_classes) + (i % out_classes)] = 1.0;
    // }

    // float * matrix = initializeFlatRandomArray(n_features, out_classes);
    // float *d_gradients;
    // float *d_inputs;
    // float *d_weights;
    // float *d_product;
    // cudaMalloc(&d_gradients, nWorkers*nThreadsPerWorker*n_features*out_classes*sizeof(float));
    // cudaMalloc(&d_inputs, test_size*n_features*sizeof(float));
    // cudaMemcpy(d_inputs, inputs, test_size*n_features*sizeof(float), cudaMemcpyHostToDevice);
    // cudaMalloc(&d_weights, n_features*out_classes*sizeof(float));
    // cudaMemcpy(d_weights, matrix, n_features*out_classes*sizeof(float), cudaMemcpyHostToDevice);
    // cudaMalloc(&d_product, BATCH_SIZE*out_classes*sizeof(float));
    // float *d_outputs = transferMatrixToDevice(outputs, test_size, out_classes);
    // int *correct = (int*)malloc(sizeof(int));
    // float *logLoss = (float*)malloc(sizeof(float));
    // correct[0] = 0;
    // logLoss[0] = 0;
    // for(int j = 0; j < epochs; j++) {
    //     correct[0] = 0;
    //     logLoss[0] = 0;
    //     for(int i = 0; i < test_size; i+=BATCH_SIZE) {
    //         predict<<<nWorkers, nThreadsPerWorker>>>(d_inputs+(i*n_features), d_weights, d_product, BATCH_SIZE, n_features, out_classes);
    //         cudaDeviceSynchronize();
    //         float * product = (float*)malloc(BATCH_SIZE*out_classes*sizeof(float));
    //         cudaMemcpy(product, d_product, BATCH_SIZE*out_classes*sizeof(float), cudaMemcpyDeviceToHost);
    //         // printf("Predicted\n");
    //         // printMatrix(product, BATCH_SIZE, out_classes);
    //         // printf("Actual\n");
    //         // printMatrix(outputs+(i*out_classes), BATCH_SIZE, out_classes);
    //         correct[0] += getAccuracy(product, outputs+(i*out_classes), BATCH_SIZE, out_classes);
    //         logLoss[0] += crossEntropyLoss(product, outputs+(i*out_classes), BATCH_SIZE, out_classes);

    //         forward_pass<<<nWorkers, nThreadsPerWorker>>>(d_inputs+(i*n_features), d_weights, d_outputs+(i*out_classes), d_product, d_gradients, BATCH_SIZE, n_features, out_classes);
    //         cudaDeviceSynchronize();
    //         // printf("Outputs\n");
    //         // printMatrix(outputs+(i*out_classes), BATCH_SIZE, out_classes);
    //         float * gradients = (float*)malloc(nWorkers*nThreadsPerWorker*n_features*out_classes*sizeof(float));
    //         cudaMemcpy(gradients, d_gradients, nWorkers*nThreadsPerWorker*n_features*out_classes*sizeof(float), cudaMemcpyDeviceToHost);
    //         // printMatrix(gradients, nWorkers*nThreadsPerWorker*n_features, out_classes);
    //         //aggregate gradients across different shards
    //         ringReduce<<<nWorkers, nThreadsPerWorker>>>(d_gradients, nThreadsPerWorker*nWorkers, n_features*out_classes, (n_features*out_classes)/(nThreadsPerWorker*nWorkers));
    //         cudaMemcpy(gradients, d_gradients, nWorkers*nThreadsPerWorker*n_features*out_classes*sizeof(float), cudaMemcpyDeviceToHost);
    //         // printf("Gradients\n");
    //         // printMatrix(gradients, n_features, out_classes);
    //         //backward pass
    //         backward_pass<<<nWorkers, nThreadsPerWorker>>>(d_weights, d_gradients, BATCH_SIZE, learning_rate, n_features, out_classes);
    //         cudaMemcpy(matrix, d_weights, n_features*out_classes*sizeof(float), cudaMemcpyDeviceToHost);
    //         // printf("Matrix\n");
    //         // printMatrix(matrix, n_features, out_classes);

    //     }
    //     float accuracy = correct[0] / (float)(test_size);
    //     std::cout << "Accuracy: "<< accuracy *100 << "%" << std::endl;
    //     std::cout << "Loss: "<< logLoss[0] << std::endl;
    // }
    return 0;
}

