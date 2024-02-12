#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

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
            //the most important line in the entire program
            weights[i][j] -= 0.5;
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

void dotProduct(int weight_h, int weight_w, int vector_h, int vector_w, float ** weights, float ** vectors, float ** product) {
    //if we have a matrix of H*W, then vector_h == weight_w
    if(weight_w != vector_h) {
        printf("INVALID VALUES FOR MATRIX AND VECTOR\n");
        return;
    }
    //initialize the matrix
    //printf("Product\n");
    for(int i = 0; i < weight_h; i++) {
        product[i] = (float*)malloc(10 * sizeof(float));
        for(int j = 0; j < vector_w; j++) {
            //printf("New entry\n");
            //printf("%d %d %f\n", i, j, weights[i][j]);
            for(int k = 0; k < weight_w; k++) {
                //printf("%f\n", weights[i][k]);
                product[i][j] += weights[i][k] * vectors[k][j];
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

void printMatrix(float ** matrix, int mat_height, int mat_width) {
    for(int i = 0; i < mat_height; i++) {
        for(int j = 0; j < mat_width; j++) {
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
    printf("%d\n", size);
    ////and we also need to cast each of the labels into a one hot vector
    float ** output;
    output = (float**)malloc(sizeof(float**) * size);
    for (int i = 0; i < size; i++) {
        output[i] = (float *)malloc(10*sizeof(float));
        output[i][labels[i]]++;
    }
    printVector(input[0], width, height);
    printVector(output[0], 10, 1);
    float **weights;
    weights = (float**)malloc(sizeof(float**)*width*height);
    initializeRandomArray(width*height, 10, weights);
    //test dotProduct function
    // float matrix[3][3] = {{1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}};
    // float vector[3][1] = {{1.0}, {2.0}, {3.0}};
    // printf("%f", matrix[2][2]);
    //the resulting matrix should be a 3x1 matrix with the values
    /*  [1.0]
        [2.0]
        [3.0]
    */
    float** product;
    product = (float**)malloc(sizeof(float**)*size);
    dotProduct(size, width * height, width * height, 10, input, weights, product);
    //printMatrix(product, 10, size);
    // dotProduct(3, 3, 3, 1, matrix, vector, product);
    // dotProduct(size, wid)
    //printf("End of execution\n");
    softmax(product, 10, 10);
    return 0;
}