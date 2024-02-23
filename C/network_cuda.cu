#include <stdio.h>
#include <stdlib.h>
#include "include/lin_alg_gpu.h"
#include <time.h>

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

float* transferMatrixToDevice(float **matrix, int height, int width) {
    float* deviceMatrix;
    cudaMalloc(&deviceMatrix, height*width*sizeof(float));
    for(int i = 0; i < height; i++) {
        cudaMemcpy(deviceMatrix+(i*width), matrix[i], sizeof(float)*width, cudaMemcpyHostToDevice);
    }
    return deviceMatrix;
}

void transferMatrixToHost(float* deviceMatrix, float** hostMatrix, int height, int width) {
    for(int i = 0; i < height; i++) {
        cudaMemcpy(hostMatrix[i], deviceMatrix+(i*width), sizeof(float)*width, cudaMemcpyHostToDevice);
    }
}

int main(void) {
    //establish important variables for training
    float learning_rate = 0.005;
    int epochs = 100;
    int BATCH_SIZE = 1000;

    srand(1);
    FILE *fptr;
    char train_data_path[] = "../MNIST_ORG/train-images.idx3-ubyte";
    char train_labels_path[] = "../MNIST_ORG/train-labels.idx1-ubyte";
    fptr = fopen(train_data_path, "rb");
    unsigned int vals[4];
    //the values are definitely stored as big endian
    int count = fread(vals,4,4, fptr);
    vals[3] = vals[3] >> 24;
    vals[2] = vals[2] >> 24;
    vals[1] = 60000;


    const int size = vals[1];
    int width = vals[2];
    int height = vals[3];

    //establish the number of thread blocks (also called n_workers) and the number of threads per block
    int n_workers = 100;
    int n_threads_per_block = 10;
    BATCH_SIZE = size / (n_threads_per_block*n_workers);
    printf("BATCH SIZE: %d\n", BATCH_SIZE);
    //read in all of the data at once using mall
    unsigned char* data;
    data = (unsigned char*)malloc(sizeof(unsigned char) * vals[1] * vals[2] * vals[3]);
    count = fread(data, sizeof(unsigned char), vals[1] * vals[2] * vals[3], fptr);
    fclose(fptr);
    //and now to try to open the training labels
    unsigned char* labels = openUByte(train_labels_path);

    //allocate everything in CPU and GPU memory
    //initialize input array in CPU memory and GPU memory
    time_t start_time = time(NULL);
    float **input;
    input = (float**) malloc(sizeof(float*) * size * width * height);
    for(int i = 0; i < size; i++) {
        input[i] = (float*)malloc(sizeof(float) * width * height);
        for(int j = 0; j < width*height; j++) {
            input[i][j] = (float) data[(i*width*height)+j] / 255;
       }
    }
    float *d_input = transferMatrixToDevice(input, size, width*height);
    free(data);

    //initialize weight array in CPU and GPU memory
    float **weights;
    weights = (float**)malloc(sizeof(float*)*width*height);
    initializeRandomArray(width*height, 10, weights);
    float *d_gradients;
    cudaMalloc(&d_gradients, n_threads_per_block*n_workers*width*height*10*sizeof(float));

    for(int i = 0; i < n_threads_per_block*n_workers; i++) {
        for(int j = 0; j < height; j++) {
            cudaMemcpy(d_gradients+(i+height*width)+(j*width), weights[j], sizeof(float)*width, cudaMemcpyHostToDevice);
        }
    }

    float *d_weights;
    d_weights = transferMatrixToDevice(weights, width*height, 10);

    //since the output one hot vectors are directly related to computing the gradients, we should pass them to the GPU as well
    float ** output;
    output = (float**)malloc(sizeof(float*) * size);
    for (int i = 0; i < size; i++) {
        output[i] = (float *)malloc(10*sizeof(float));
        for(int j = 0; j < 10; j++) {
            output[i][j] = 0;
        }
        output[i][labels[i]] = 1.0;
    }
    float *d_outputs = transferMatrixToDevice(output, size, 10);
    time_t end_time = time(NULL);
    double diff = difftime(end_time, start_time);
    time_t start_train_time = time(NULL);
    int * counter = 0;
    cudaMemcpy(&counter, 0, sizeof(int), cudaMemcpyHostToDevice);
    // printf("Finished memory allocation in %f seconds\n", diff);
    // //forward pass
    forward_pass<<<n_workers, n_threads_per_block>>>(BATCH_SIZE, d_input, d_weights, d_outputs, d_gradients, size, width*height, 10, counter);
    cudaDeviceSynchronize();
    // time_t end_train_time = time(NULL);
    // diff = difftime(end_train_time, start_train_time);
    // printf("Finished forward pass in %f seconds\n", diff);
    //pass weights back to CPU and verify correctness
    for(int i = 0; i < n_threads_per_block*n_workers; i++) {
        transferMatrixToHost(d_gradients+(i+height*width), weights, width*height, 10);
        printf("Batch %d\n", i);
        printMatrix(weights, BATCH_SIZE, 10);
    }
    //Before we start training, we need to
    //allocate n_workers copies of the weight matrix to each different thread

    /*1. "In synchronous training, the forward pass begins at the same time in all of the workers and they compute a different output and gradients. 
    Here each worker waits for all other workers to complete their training loops and calculate their respective gradients"
    2. After all gradients have been calculated, we aggregate the gradients using the all-reduce algorithm (we will compute all reduce and multiply the combined gradient by the learning rate on the host/CPU side)
    3. Send a copy of the updated gradients to all of the client workers then continue backprop on their end as normal
    */
    cudaFree(d_input);
  	cudaFree(d_outputs);
    cudaFree(d_weights);
    cudaFree(d_gradients);
    free(weights);
    free(input);
    return 0;
}