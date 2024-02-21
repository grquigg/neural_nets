#include <stdio.h>
#include <stdlib.h>


__global__
void normalize(float *input, int row_size) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    // printf("Valid\n");
    for(int index = 0; index < row_size; index++) {
        input[i*row_size + index] = 1.0;
    }
}

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


void forward_pass() {}

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
    int n_workers = 1000;
    int n_threads_per_block = 1;
    
    //read in all of the data at once using mall
    unsigned char* data;
    data = (unsigned char*)malloc(sizeof(unsigned char) * vals[1] * vals[2] * vals[3]);
    count = fread(data, sizeof(unsigned char), vals[1] * vals[2] * vals[3], fptr);
    fclose(fptr);
    //and now to try to open the training labels
    unsigned char* labels = openUByte(train_labels_path);

    //every piece of memory that we store needs to be matched by a block of memory of equal size in the array
    float *d_input;
    cudaMalloc(&d_input, size*width*height*sizeof(float));
    float **input;
    input = (float**) malloc(sizeof(float*) * size * width * height);
    for(int i = 0; i < size; i++) {
        input[i] = (float*)malloc(sizeof(float) * width * height);
        for(int j = 0; j < width*height; j++) {
            input[i][j] = (float) data[(i*width*height)+j] / 255;
       }
       cudaMemcpy(d_input+(i*width*height), input[i], sizeof(float)*width*height, cudaMemcpyHostToDevice);
    }

    cudaMemcpy(input[0], d_input, sizeof(float)*width*height, cudaMemcpyDeviceToHost);
    printMatrix(input, 1, width*height);
    //Before we start training, we need to
    //allocate n_workers copies of the weight matrix to each different thread

    /*1. "In synchronous training, the forward pass begins at the same time in all of the workers and they compute a different output and gradients. 
    Here each worker waits for all other workers to complete their training loops and calculate their respective gradients"
    2. After all gradients have been calculated, we aggregate the gradients using the all-reduce algorithm (we will compute all reduce and multiply the combined gradient by the learning rate on the host/CPU side)
    3. Send a copy of the updated gradients to all of the client workers then continue backprop on their end as normal
    */
    return 0;
}