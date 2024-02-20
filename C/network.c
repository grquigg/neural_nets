#define OVERFLOW_FLAG 1.175494351e-33F
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "include/lin_alg.h"
#include <stdbool.h>

unsigned char * openUByte(char * path) {
    FILE *ptr;
    ptr = fopen(path, "rb");
    int magic, size;
    fread(&magic, 4, 1, ptr);
    fread(&size, 4, 1, ptr);
    unsigned char *labels;
    labels = (unsigned char *)malloc(sizeof(unsigned char)*size);
    fread(labels, sizeof(unsigned char), size, ptr);
    fclose(ptr);
    return labels;
}

void initializeRandomArray(int height, int width, float ** weights) {
    int a = 1;
    for (int i = 0; i < height; i++) {
        weights[i] = (float*)malloc(sizeof(float) * width);
        for(int j = 0; j < width; j++) {
            weights[i][j] = (float)rand()/(float)(RAND_MAX/a);
            // weights[i][j] = 0.5;
            //the most important line in the entire program
        }
    }
}

void softmax(float** product, int product_height, int product_width) {
    float total = 0.0;
    float logSumTotal = 0.0;
    for (int i = 0; i < product_height; i++) {
        total = 0.0;
        for (int j = 0; j < product_width; j++) {
            total += exp(product[i][j]);
        }
        logSumTotal = logf(total);
        float prob_sums = 0.0;
        for (int j = 0; j < product_width; j++) {
            product[i][j] = exp(product[i][j] - logSumTotal);
            prob_sums += product[i][j];
        }
        
    }
}

float crossEntropyLoss(float** loss, int matrix_height, int matrix_width, float **groundTruth) {
    float log_sum = 0;
    for (int i = 0; i < matrix_height; i++) {
        for (int j = 0; j < matrix_width; j++) {
            if (loss[i][j] != 0.0) {
                log_sum -= logf(fabsf(loss[i][j]));
            }
        }
    }
    return log_sum;
}
void printMatrix(float ** matrix, int mat_height, int mat_width) {
    // printf("[");
    for(int i = 0; i < mat_height; i++) {
        // printf("[");
        for(int j = 0; j < mat_width; j++) {
            printf("%f\t", matrix[i][j]);
        }
        printf("\n");
    }
    // printf("]\n");
}

int computeCorrect(float** output, int output_h, int output_w, float** predicted) {
    int correct = 0;
    for (int i = 0; i < output_h; i++) {
        int max = 0;
        float max_score = 0.0;
        int actual = 0;
        for (int j = 0; j < output_w; j++) {
            if (predicted[i][j] > max_score) {
                max = j;
                max_score = predicted[i][j];
            }
            // printf("Max for %d: %d\n", i, max);
            // printf("Output at %d %d: %f\n", i, j, output[i][j]);
            if (output[i][j] == 1.0) {
                actual = j;
            }
        }
        if ((int) actual == max) correct++;
    }
    return correct;
}

void assert(bool condition) {
    if(!condition) {
        printf("ASSERTION FAILED\n");
    }
}

// void testFunctions(float learning_rate, int epochs) {
// //    EXAMPLES FOR TESTING
//     float** test_input;
//     int test_height = 4;
//     int test_features = 4;
//     int test_classes = 5;
//     test_input = (float**)malloc(sizeof(float**) * test_height);
//     initializeRandomArray(test_height, test_features, test_input);
//     printf("Test inputs\n");
//     printMatrix(test_input, test_height, test_features);
//     float** test_weights;
//     // printf("Test weights\n");
//     test_weights = (float**)malloc(sizeof(float**)*test_features);
//     initializeRandomArray(test_features, test_classes, test_weights);
//     printf("Original weights\n");
//     printMatrix(test_weights, test_features, test_classes);
//     float** test_output;
//     test_output = (float**)malloc(sizeof(float**) * test_height);
//     for (int i = 0; i < test_height; i++) {
//         test_output[i] = (float*)malloc(sizeof(float) * test_classes);
//         for (int j = 0; j < test_classes; j++) {
//             test_output[i][j] = 0.0;
//         }
//         test_output[i][i % test_classes] = 1.0;
//     }
//     printMatrix(test_output, test_height, test_classes);
//     float** product;
//     product = (float**)malloc(sizeof(float**) * test_features);
//     for (int i = 0; i < epochs; i++) {
//         printf("Epoch %d\n", i+1);
//         dotProduct(test_height, test_features, test_features, test_classes, test_input, test_weights, product);
//         softmax(product, test_height, test_classes);
//         printf("Actual\n");
//         printMatrix(product, test_height, test_classes);
//         int correct = computeCorrect(test_output, test_height, test_classes, product);
//         float accuracy = correct / (float) test_height;
//         printf("Accuracy: %f%%\n", accuracy * 100);
//         //actual - expected
//         matrixSubtract(product, test_output, test_height, test_classes, test_height, test_classes, -1);
//         printf("Actual - expected\n");
//         printMatrix(product, test_height, test_classes);
//         float loss = crossEntropyLoss(product, test_height, test_classes, test_output);
//         printf("\nLoss: %f\n", loss);
//         //and here's the part where we update only relevant weights
//         //assert that the dimensions are equal
//         // matrixMultiply(product, test_output, test_height, test_classes);
//         // printMatrix(product, test_height, test_classes);
//         float** updated_weights;
//         updated_weights = (float**)malloc(sizeof(float**) * test_height);
//         dotProduct(test_features, test_height, test_height, test_classes, transposeMatrix(test_input, test_height, test_features), product, updated_weights);
//         printf("Update:\n");
//         printMatrix(updated_weights, test_features, test_classes);
//         multiplyMatrixByScalar(updated_weights, test_features, test_classes, learning_rate);
//         // matrixSubtract(test_weights, updated_weights, test_features, test_classes, test_features, test_classes, 1);
//         matrixAdd(test_weights, updated_weights, test_features, test_classes, test_features, test_classes);
//         printf("Weights after update:\n");
//         printMatrix(test_weights, test_features, test_classes);
//     }
// }

int main(void) {
    srand(1);
    FILE *fptr;
    char train_data_path[] = "../mnist/train-images.idx3-ubyte";
    char train_labels_path[] = "../mnist/train-labels.idx1-ubyte";
    fptr = fopen(train_data_path, "rb");
    unsigned int vals[4];
    float learning_rate = 0.005;
    int epochs = 100;
    //the values are definitely stored as big endian
    int count = fread(vals,4,4, fptr);
    // Print the file content
    vals[3] = vals[3] >> 24;
    vals[2] = vals[2] >> 24;
    vals[1] = 60000;
    const int size = vals[1];
    int width = vals[2];
    int height = vals[3];
    //read in all of the data at once using mall
    unsigned char* data;
    data = (unsigned char*)malloc(sizeof(unsigned char) * vals[1] * vals[2] * vals[3]);
    count = fread(data, sizeof(unsigned char), vals[1] * vals[2] * vals[3], fptr);
    fclose(fptr);
    //and now to try to open the training labels
    printf("outputs\n");
    unsigned char* labels = openUByte(train_labels_path);

    int BATCH_SIZE = 1000;
    //all of the data is stored! the next thing that needs to be done is to convert the image data and
    //the label data into floats and ints, respectively
    //we're not going to have negative pixel values but we still need the precision for the linear algebra that we're
    ////going to need to be doing
    ////we will most likely need an array of float pointers for this in order to do the math quickly
    float **input;
    input = (float**) malloc(sizeof(float**) * size);
    //
    //printf("This is a test\n");
    for(int i = 0; i < size; i++) {
        input[i] = (float*)malloc(sizeof(float) * width * height);
        for(int j = 0; j < width*height; j++) {
            input[i][j] = (float) data[(i*width*height)+j] / 255;
       }
    }
    free(data);
    ////and we also need to cast each of the labels into a one hot vector
    float ** output;
    output = (float**)malloc(sizeof(float**) * size);
    for (int i = 0; i < size; i++) {
        output[i] = (float *)malloc(10*sizeof(float));
        for(int j = 0; j < 10; j++) {
            output[i][j] = 0;
        }
        output[i][labels[i]] = 1.0;
    }
    float **weights;
    weights = (float**)malloc(sizeof(float**)*width*height);
    initializeRandomArray(width*height, 10, weights);
    // testFunctions(learning_rate, epochs);
    float** product;
    product = (float**)malloc(sizeof(float**) * BATCH_SIZE);
    for (int i = 0; i < epochs; i++) {
        float accuracy = 0.0;
        int totalCorrect = 0;
        for (int j = 0; j < size; j += BATCH_SIZE) {
            //forward pass
            dotProduct(BATCH_SIZE, width * height, width * height, 10, input+j, weights, product);
            softmax(product, BATCH_SIZE, 10);
            //calculate accuracy
            totalCorrect += computeCorrect(output + (j), BATCH_SIZE, 10, product);
            accuracy = totalCorrect/ (float) (j + BATCH_SIZE);
            matrixSubtract(product, output + (j), BATCH_SIZE, 10, BATCH_SIZE, 10, -1);

            float loss = crossEntropyLoss(product, BATCH_SIZE, 10, output+j);
            float** updated_weights;

            //backprop
            updated_weights = (float**)malloc(sizeof(float**) * width * height);
            dotProduct(width * height, BATCH_SIZE, BATCH_SIZE, 10, transposeMatrix(input+j, BATCH_SIZE, width*height), product, updated_weights);
            multiplyMatrixByScalar(updated_weights, width * height, 10, learning_rate*(1.0/(float) BATCH_SIZE));
            matrixAdd(weights, updated_weights, width * height, 10, width * height, 10);
        }
        printf("end of epoch %d\n", i + 1);
        printf("accuracy: %f%%\n", accuracy * 100);
    }
    return 0;
}