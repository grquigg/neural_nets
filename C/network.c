#define OVERFLOW_FLAG 1.175494351e-33F
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
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
            // weights[i][j] = (float)rand()/(float)(RAND_MAX/a);
            weights[i][j] = 0.0;
            //the most important line in the entire program
        }
    }
}

void printVector(float * vec, int vec_height, int vec_width) {
    for(int i = 0; i < vec_width; i++) {
        for(int j = 0; j < vec_height; j++) {
            printf("%.2f\t", vec[(i*vec_width)+j]);
        }
        printf("\n");
    }
}

void dotProductWithTranspose(int weight_h, int weight_w, int vector_h, int vector_w, float** weights, float** vectors, float** product) {
    //whichever matrix is given first is the one that will be transposed
    //resulting matrix will be of size weight_w, vector_w
    //"row"
    for (int row = 0; row < weight_w; row++) {
        product[row] = (float*)malloc(vector_w * sizeof(float));
        for (int col = 0; col < vector_w; col++) {
            product[row][col] = 0;
            for (int j = 0; j < vector_h; j++) {
                if (weights[j][row] < OVERFLOW_FLAG) {
                    continue;
                }
                product[row][col] += weights[j][row] * vectors[j][col];
                //printf("product at %d %d: %.15f\n", row, col, product[row][col]);
                if (product[row][col] > 10000) {
                    printf("%d %d %d\n", row, col, j);
                    printf("problem children: %.15f %.15f %.15f\n", weights[j][row], vectors[j][col], weights[j][row] * vectors[j][col]);
                    printf("product at %d %d: %f\n", row, col, product[row][col]);
                }
            }
            //printf("Answer at %d %d: %f\n", row, col, product[row][col]);
        }
    }
}

void dotProduct(int weight_h, int weight_w, int vector_h, int vector_w, float ** weights, float ** vectors, float ** product, bool inverse) {
    //if we have a matrix of H*W, then vector_h == weight_w
    if (inverse) {
        dotProductWithTranspose(weight_h, weight_w, vector_h, vector_w, weights, vectors, product);
        return;
    }
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

void dotProductFromSubset(int mat1_begin, int mat1_end, int weight_w, int vector_h, int vector_w, float** weights, float** vectors, float** product, bool inverse) {
    int mat_h = mat1_end - mat1_begin;
    for (int i = mat1_begin; i < mat1_end; i++) {
        product[i] = (float*)malloc(vector_w * sizeof(float));
        //printf("Success %d\n", i);
        for (int j = 0; j < vector_w; j++) {
            product[i][j] = 0;
            //printf("New entry\n");
            //printf("%d %d %f\n", i, j, weights[i][j]);
            for (int k = 0; k < weight_w; k++) {
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

void matrixSubtract(float ** matrix1, float **matrix2, float m1_h, float m1_w, float m2_h, float m2_w, float scalar) {
    if (m1_h == m2_h && m1_w == m2_w) {
        for (int i = 0; i < m1_h; i++) {
            for (int j = 0; j < m1_w; j++) {
                matrix1[i][j]-=matrix2[i][j];
                if(matrix1[i][j] < OVERFLOW_FLAG) {
                    matrix1[i][j] = 0;
                }
                else {
                    matrix1[i][j] *= scalar;
                }
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

float crossEntropyLoss(float** loss, int matrix_height, int matrix_width) {
    float log_sum = 0;
    for (int i = 0; i < matrix_height; i++) {
        for (int j = 0; j < matrix_width; j++) {
            if (loss[i][j] != 0) {
                printf("%f %f\n", log_sum, fabsf(loss[i][j]));
                log_sum -= logf(fabsf(loss[i][j]));
            }
        }
    }
    return log_sum;
}
void printMatrix(float ** matrix, int mat_height, int mat_width) {
    for(int i = 0; i < mat_height; i++) {;
        for(int j = 0; j < mat_width; j++) {
            printf("%f\t", matrix[i][j]);
        }
        printf("\n");
    }
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
            if (output[i][j] == 1) {
                actual = j;
            }
        }
        if (actual == max) correct++;
    }
    return correct;
}

void testFunctions(float learning_rate, int epochs) {
//    EXAMPLES FOR TESTING
    float** test_input;
    test_input = (float**)malloc(sizeof(float**) * 100);
    initializeRandomArray(100, 3, test_input);
    printf("Test inputs\n");
    printMatrix(test_input, 100, 3);
    float** test_weights;
    printf("Test weights\n");
    test_weights = (float**)malloc(sizeof(float**)*3);
    initializeRandomArray(3, 5, test_weights);
    printMatrix(test_weights, 3, 5);
    float** test_output;
    test_output = (float**)malloc(sizeof(float**) * 100);
    for (int i = 0; i < 100; i++) {
        test_output[i] = (float*)malloc(sizeof(float) * 5);
        for (int j = 0; j < 5; j++) {
            test_output[i][j] = 0.0;
        }
        test_output[i][i % 4] = 1.0;
    }
    float** product;
    product = (float**)malloc(sizeof(float**) * 100);
    for (int i = 0; i < epochs; i++) {
        printf("Epoch %d\n", i+1);
        dotProduct(100, 3, 3, 5, test_input, test_weights, product, false);
        printMatrix(product, 100, 5);
        softmax(product, 100, 5);
        printf("Product after softmax\n");
        printMatrix(product, 100, 5);
        printMatrix(test_output, 100, 5);
        float accuracy = computeCorrect(test_output, 100, 5, product);
        printf("Accuracy: %f%%\n", accuracy * 100);
        matrixSubtract(product, test_output, 100, 5, 100, 5, -1);
        printf("Product after getting the loss:\n");
        printMatrix(product, 100, 5);
        float loss = crossEntropyLoss(product, 100, 5);
        printf("Loss: %f\n", loss);
        float** updated_weights;
        updated_weights = (float**)malloc(sizeof(float**) * 100 * 5);
        dotProduct(100, 3, 100, 5, test_input, product, updated_weights, true);
        printf("Update:\n");
        printMatrix(updated_weights, 3, 5);
        multiplyMatrixByScalar(updated_weights, 3, 5, learning_rate);
        printf("New\n");
        printMatrix(updated_weights, 3, 5);
        matrixSubtract(test_weights, updated_weights, 3, 5, 3, 5, 1);
        printMatrix(test_weights, 3, 5);
    }
}
int main(void) {
    FILE *fptr;
    char train_data_path[] = "../mnist/train-images.idx3-ubyte";
    char train_labels_path[] = "../mnist/train-labels.idx1-ubyte";
    fptr = fopen(train_data_path, "rb");
    unsigned int vals[4];
    float learning_rate = 0.0005;
    int epochs = 1;
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
        output[i][labels[i]]++;
    }
    float **weights;
    weights = (float**)malloc(sizeof(float**)*width*height);
    initializeRandomArray(width*height, 10, weights);
    //testFunctions(learning_rate, epochs);
    float** product;
    product = (float**)malloc(sizeof(float**) * BATCH_SIZE);
    for (int i = 0; i < epochs; i++) {
        float accuracy = 0.0;
        int totalCorrect = 0;
        for (int j = 0; j < size; j += BATCH_SIZE) {
            //forward pass
            printf("Batch starting at %d\n", j);
            dotProduct(BATCH_SIZE, width * height, width * height, 10, input+j, weights, product, false);
            //printMatrix(product, BATCH_SIZE, 10);
            softmax(product, BATCH_SIZE, 10);
            printf("New product\n");
            printMatrix(product, BATCH_SIZE, 10);
            //calculate accuracy
            totalCorrect += computeCorrect(output + (j), BATCH_SIZE, 10, product);
            accuracy = totalCorrect/ (j + BATCH_SIZE);
            printf("accuracy: %f%%\n", accuracy * 100);
            printMatrix(output + j, BATCH_SIZE, 10);
            matrixSubtract(product, output + (j), BATCH_SIZE, 10, BATCH_SIZE, 10, 1);
            printMatrix(product, BATCH_SIZE, 10);
            // float loss = crossEntropyLoss(product, BATCH_SIZE, 10);
            // printf("loss: %f\n", loss);
            // float** updated_weights;
            // updated_weights = (float**)malloc(sizeof(float**) * width * height);
            // //printf("Reverse dot product\n");
            // //printMatrix(product, BATCH_SIZE, 10);
            // dotProduct(BATCH_SIZE, width * height, BATCH_SIZE, 10, input+j, product, updated_weights, true);
            // //printf("Weight updates\n");
            // multiplyMatrixByScalar(updated_weights, width * height, 10, learning_rate);
            // matrixAdd(weights, updated_weights, width * height, 10, width * height, 10);
            //printMatrix(weights, width * height, 10);
        }
        printf("end of epoch %d\n", i + 1);
    }
    return 0;
}