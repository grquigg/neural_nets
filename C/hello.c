#include <stdio.h>
#include <stdlib.h>
#include <math.h>

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
        weights[i] = (float *)malloc(sizeof(float)*width);
        for(int j = 0; j < width; j++) {
            weights[i][j] = (float)rand()/(float)(RAND_MAX/a);
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

void returnProduct(float ** weights, float **vectors, float weight_h, float weight_w, float vector_h, float vector_w) {

}

void printMatrix(float ** matrix, int mat_height, int mat_width) {
    for(int i = 0; i < mat_width; i++) {
        for(int j = 0; j < mat_height; j++) {
            printf("%.2f\t", matrix[i][j]);
        }
        printf("\n");
    }
}

int main(void) {
    FILE *fptr;
    char train_data_path[] = "../mnist/train-images.idx3-ubyte";
    char train_labels_path[] = "../mnist/train-labels.idx1-ubyte";
    fptr = fopen(train_data_path, "rb");
    unsigned int vals[4];

    //the values are definitely stored as big endian
    int count = fread(vals,4,4, fptr);
    // Print the file content
    vals[3] = vals[3] >> 24;
    vals[2] = vals[2] >> 24;
    vals[1] = 60000;
    int size = vals[1];
    int width = vals[2];
    int height = vals[3];
    //read in all of the data at once using malloc
    unsigned char *data;
    data = (unsigned char *)malloc(sizeof(unsigned char)*vals[1]*vals[2]*vals[3]);
    count = fread(data, sizeof(unsigned char), vals[1]*vals[2]*vals[3], fptr);
    fclose(fptr);
    //and now to try to open the training labels
    unsigned char *labels = openUByte(train_labels_path);

    //all of the data is stored! the next thing that needs to be done is to convert the image data and
    //the label data into floats and ints, respectively
    //we're not going to have negative pixel values but we still need the precision for the linear algebra that we're
    //going to need to be doing
    //we will most likely need an array of float pointers for this in order to do the math quickly
    float * input[size];
    for(int i = 0; i < size; i++) {
        input[i] = (float *)malloc(width*height*sizeof(float));
        for(int j = 0; j < width*height; j++) {
            input[i][j] = (float) data[(i*width*height)+j] / 255;
        }
    }
    free(data);
    //and we also need to cast each of the labels into a one hot vector
    float * output[size];
    for (int i = 0; i < size; i++) {
        output[i] = (float *)malloc(10*sizeof(float));
        output[i][labels[i]]++;
    }
    printVector(input[0], width, height);
    printVector(output[0], 10, 1);
    float * weights[height];
    initializeRandomArray(width*height, 1, weights);
    float **product;
    // printMatrix(weights, 1, width*height);
    return 0;
}