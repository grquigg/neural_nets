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
        logSumTotal = log(total);
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
            product[i*weight_w+j] = 0.0;
            for(int k = 0; k < vector_w; k++) { //we compute the kth entry in row i of the INPUTS times the kth entry in column j of the WEIGHTS
                product[i*weight_w+j] += inputs[i*vector_w+k] * weights[k*weight_w+j];
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

__device__ void matrixSubtract(float * matrix1, float *matrix2, int m1_h, int m1_w, int m2_h, int m2_w, float* outVec) {
    if (m1_h == m2_h && m1_w == m2_w) {
        for (int i = 0; i < m1_h; i++) {
            for (int j = 0; j < m1_w; j++) {
                outVec[(i*m1_w)+j] = matrix1[(i*m1_w)+j]-matrix2[(i*m1_w)+j];
            }
        }
    }
}

__device__ void matrixAdd(float * matrix1, float * matrix2, int m1_h, int m1_w) {
    for(int i = 0; i < m1_h; i++) {
        for(int j = 0; j < m1_w; j++) {
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
    int index = blockIdx.x*blockDim.x + threadIdx.x;
    int begin_part = index*chunk_size;
    int end_part = (index+1)*chunk_size;
    for(int i = 1; i < total_steps; i++) {
        for(int j = begin_part; j < end_part; j++) {
            gradients[j] += gradients[(i*step_size)+j];
        }
    }
}

__global__ void ringReduce(LogisticRegression * model, const int total_steps, const int step_size, const int chunk_size) {
    int index = blockIdx.x*blockDim.x + threadIdx.x;
    int begin_part = index*chunk_size;
    int end_part = (index+1)*chunk_size;
    for(int i = 1; i < total_steps; i++) {
        for(int j = begin_part; j < end_part; j++) {
            model->gradients[j] += model->gradients[(i*step_size)+j];
        }
    }
    // printf("Ring reduce\n");
}
__global__ void predict(float * inputs, float* weights, float * product, int size, int n_features, int n_classes) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    int batch = size / (blockDim.x * gridDim.x);
    dotProduct(inputs+(i*n_features*batch), weights, product+(i*n_classes*batch), batch, n_features, n_features, n_classes);
    softmax(product+(i*n_classes*batch), batch, n_classes);
}

__global__ void predict(LogisticRegression* model, float* inputs, float* product, int size) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    int batch = size / (blockDim.x * gridDim.x);
    dotProduct(inputs+(i*(model->nFeatures)*batch), model->weights, product+(i*(model->nClasses)*batch), batch, model->nFeatures, model->nFeatures, model->nClasses);
    softmax(product+(i*(model->nClasses)*batch), batch, (model->nClasses));
}
__global__ void forward_pass(float* inputs, float* weights, float* outputs, float* product, float* gradients, int size, int n_features, int n_classes) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    int batch = size / (blockDim.x * gridDim.x);
    matrixSubtract(product+(i*n_classes*batch), outputs+(i*n_classes*batch), batch, n_classes, batch, n_classes, product+(i*n_classes*batch));
    dotProductTranspose(inputs+(i*batch*n_features), product+(i*batch*n_classes), gradients+(i*n_features*n_classes), batch, n_features, batch, n_classes);
}

__global__ void forward_pass(LogisticRegression* model, float* inputs, float* outputs, float* product, int size, int nClasses) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    int batch = size / (blockDim.x * gridDim.x);
    matrixSubtract(product+(i*nClasses*batch), outputs+(i*nClasses*batch), batch, nClasses, batch, nClasses, product+(i*nClasses*batch));
    dotProductTranspose(inputs+(i*batch*(model->nFeatures)), product+(i*batch*(model->nClasses)), ((*model).gradients)+(i*(model->nClasses)*(model->nFeatures)), batch, (model->nFeatures), batch, (model->nClasses));
}

__global__ void backward_pass(float* weights, float * gradients, int batch_size, float learning_rate, int n_features, int n_classes) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    //BATCH_SIZE*n_classes length vector
    int batch = n_features / (blockDim.x * gridDim.x);
    matrixMultiplyByScalar(gradients+(i*batch*n_classes), batch, n_classes, learning_rate/(float) batch_size);
    matrixSubtract(weights+(i*n_classes*batch), gradients+(i*n_classes*batch), batch, n_classes, batch, n_classes, weights+(i*n_classes*batch));
}

__global__ void backward_pass(LogisticRegression* model, int batch_size, float learning_rate) {
    int index = blockIdx.x*blockDim.x + threadIdx.x;
    //BATCH_SIZE*n_classes length vector
    int batch = model->nFeatures / (blockDim.x * gridDim.x);
    int start = index*batch*(model->nClasses);
    for(int i = 0; i < batch; i++) {
        for(int j = 0; j < model->nClasses; j++) {
            (*model).gradients[start+i*(model->nClasses)+j] *= (learning_rate / batch_size);
            (*model).weights[start+i*(model->nClasses)+j] -=  (*model).gradients[start+i*(model->nClasses)+j];
        }
    }
    // printf("Finish backward\n");
}