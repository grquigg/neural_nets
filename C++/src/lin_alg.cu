#include <vector>
#include <iostream>
#include "../include/lin_alg.h"

//////////DEVICES////////
__device__ void softmax(float* product, int product_height, int product_width) {
    float total = 0.0;
    float logSumTotal = 0.0;
    for (int i = 0; i < product_height; i++) {
        total = 0.0;
        for (int j = 0; j < product_width; j++) {
            total += exp(product[i*product_width+j]);
        }
        logSumTotal = logf(total);
        float prob_sums = 0.0;
        for (int j = 0; j < product_width; j++) {
            product[i*product_width+j] = exp(product[i*product_width+j] - logSumTotal);
            prob_sums += product[i*product_width+j];
        }
        
    }
}
__device__ void dotProduct(float* inputs, float* weights, float * product, int vector_h, int vector_w, int weight_h, int weight_w) {
    //initialize the matrix
    //dot product is ALWAYS computed as the rows of the first matrix by the columns of the second matrix
    for(int i = 0; i < vector_h; i++) { //for every row in the first matrix
        for(int j = 0; j < weight_w; j++) { //for every column in the second matrix
            product[i*weight_w+j] = 0.0f;
            for(int k = 0; k < vector_w; k++) { //we compute the kth entry in row i of the INPUTS times the kth entry in column j of the WEIGHTS
                product[i*weight_w+j] += inputs[i*vector_w+k] * weights[k*weight_w+j];
                if (product[i*vector_w+j] > 10000) {
                    // printf("Problem children from normal dot product: %.15f %.15f\n", weights[i][k], vectors[k][j]);
                    // printf("Product at %d %d: %f\n", i, j, product[i][j]);
                }
                //printf("Temp product: %f\n", product[i][j]);
            }
            //printf("%f\n", product[i][j]);
        }
    }
}

__device__ void dotProductTranspose(float* inputs, float* weights, float * product, int vector_h, int vector_w, int weight_h, int weight_w) {
    //remember that we want the resulting matrix to be of shape [vector_h, weight_w]
    for(int i = 0; i < vector_w; i++) {
        for(int j = 0; j < weight_w; j++) {
            product[i*weight_w+j] = 0.0;
            for(int k = 0; k < vector_h; k++) {
                product[i*weight_w+j] += inputs[k*vector_w+i] * weights[k*weight_w+j];
            }
        }
    }
}

__device__ void matrixSubtract(float * matrix1, float *matrix2, int m1_h, int m1_w, int m2_h, int m2_w, float scalar) {
    if (m1_h == m2_h && m1_w == m2_w) {
        for (int i = 0; i < m1_h; i++) {
            for (int j = 0; j < m1_w; j++) {
                matrix1[(i*m1_w)+j]-=matrix2[(i*m1_w)+j];
                matrix1[(i*m1_w)+j]*=-1;
            }
        }
    }
}

__device__ void matrixAdd(float * matrix1, float * matrix2, int m1_h, int m1_w) {
    for(int i = 0; i < m1_h; i++) {
        for(int j = 0; j < m1_h; j++) {
            matrix1[i*m1_w+j] += matrix2[i*m1_w+j];
        }
    }
}

__device__ void matrixMultiplyByScalar(float* mat, int m1_h, int m1_w, float scalar) {
    for(int i = 0; i < m1_h; i++) {
        for(int j = 0; j < m1_w; j++) {
            mat[(i*m1_w)+j]*= scalar;
        }
    }
}

/////METRIC COMPUTATIONS/////
__device__ int getAccuracy(float* predicted, float* actual, int height, int width) {
    int correct = 0;
    for (int i = 0; i < height; i++) {
        int max = 0;
        float max_score = 0.0;
        int a = 0;
        for (int j = 0; j < width; j++) {
            if (predicted[(i*width)+j] > max_score) {
                max = j;
                max_score = predicted[(i*width)+j];
            }
            if (actual[(i*width)+j] == 1.0) {
                a = j;
            }
        }
        if ((int) a == max) correct++;
    }
    return correct;
}

//////////GLOBALS////////
/*
*gradients: the gradient vector
begin_part: the index of where this threads cumulative gradients begin in the subsection
end_part: the index of where this threads cumulative gradients end in the subsection
total_steps: the total number of steps that each thread needs to perform in order to acheive the full cumulative gradient
step_size: the total distance in grads that we need to jump every time
*/
__global__ void ringReduce(float* gradients, const int total_steps, const int step_size, const int chunk_size) {
    //we achieve our reduction in two loops: update and set
    //in the update loop, we're simply calculating the cumulative sum of each part of the respective gradients
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    int begin_part = i*chunk_size;
    int end_part = (i+1)*chunk_size;
    for(int i = 1; i < total_steps; i++) {
        for(int j = begin_part; j < end_part; j++) {
            gradients[j] += gradients[(i*step_size)+j];
        }
    }
    for(int i = 0; i < total_steps-1; i++) {
        for(int j = 0; j < step_size; j++) {
            gradients[(i*step_size)+j] = gradients[j] / total_steps;
        }
    }
    //and in the set loop, we're setting every copy of the gradients in our array to be equal to the most recently updated entry
}

__global__ void forward_pass(float* inputs, float* weights, float* outputs, float* product, float* gradients, int size, int n_features, int n_classes, int* correct, float * loss) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    //BATCH_SIZE*n_classes length vector
    dotProduct(inputs+(i*size*n_features), weights, product+(i*size*n_classes), size, n_features, n_features, n_classes);
    softmax(product+(i*size*n_classes), size, n_classes);
    (*correct) += getAccuracy(product+(i*size*n_classes), outputs+(i*size*n_classes), size, n_classes);
    //we can compute accuracy on the forward pass
    matrixSubtract(product+(i*size*n_classes), outputs+(i*size*n_classes), size, n_classes, size, n_classes, -1);
    dotProductTranspose(inputs+(i*size*n_features), product+(i*size*n_classes), gradients+(i*n_features*n_classes), size, n_features, size, n_classes);
    //ring reduce
    //we can allow each thread to 
    //index i means that we are responsible for cumulating together the ith chunk of gradient
    // ringReduce(gradients, i*chunk_size, (i+1)*chunk_size, total_parts, n_features*n_classes, chunk_size);
}

__global__ void backward_pass(float* weights, float * gradients, int batch_size, float learning_rate, int n_features, int n_classes) {
    matrixMultiplyByScalar(gradients, n_features, n_classes, learning_rate/batch_size);
    matrixAdd(weights, gradients, n_features, n_classes);
}