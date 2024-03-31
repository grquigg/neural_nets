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

__device__ void sigmoid(float* inputs, int size) {
    for(int i = 0; i < size; i++) {
        inputs[i] = (1/ (1+expf(-inputs[i])));
    }
}
__global__ void sigmoidD(float* activations, int height, int width, float * delta) {
    if((height * width) % (gridDim.x * blockDim.x) != 0) {
        printf("Bad outcome\n");
    }
    printf("Sigmoid\n");
    int batch = (height * width) / (gridDim.x * blockDim.x);
    int index = blockIdx.x*blockDim.x + threadIdx.x;
    for(int i = 0; i < batch; i++) {
        delta[index*batch+i] *= activations[index*batch+i] * (1-activations[index*batch+i]);
    }
}
__device__ float* transposeMatrix(float * matrix, int matrix_height, int matrix_width) {
    float * transpose = new float[matrix_width*matrix_height];
    for(int i = 0; i < matrix_height; i++) {
        for(int j = 0; j < matrix_width; j++) {
            transpose[i*matrix_width+j] = matrix[j*matrix_height+i];
            // printf("Valid %d %d %f %f\n", i, j, transpose[i*matrix_width+j], matrix[j*matrix_height+i]);
        }
    }
    return transpose;
}

__device__ void dotProduct(float* inputs, float* weights, float * product, int vector_h, int vector_w, int weight_h, int weight_w) {
    //initialize the matrix
    //dot product is ALWAYS computed as the rows of the first matrix by the columns of the second matrix
    if (vector_w != weight_h) {
        printf("invalid values\n");
        return;
    }
    for(int i = 0; i < vector_h; i++) { //for every row in the first matrix
        for(int j = 0; j < weight_w; j++) { //for every column in the second matrix
            product[i*weight_w+j] = 0.0;
            for(int k = 0; k < vector_w; k++) { //we compute the kth entry in row i of the INPUTS times the kth entry in column j of the WEIGHTS
                product[i*weight_w+j] += inputs[i*vector_w+k] * weights[k*weight_w+j];
                // printf("This %d %d %f %f\n", i, j, inputs[i*vector_w+k], weights[k*weight_w+j]);
            }
            printf("%f\n", product[i*weight_w+j]);
        }
    }
}

__device__ void dotProduct(float* inputs, float* weights, float * product, int vector_h, int vector_w, int weight_h, int weight_w, float* bias) {
    if (vector_w != weight_h) {
        printf("invalid values\n");
        return;
    }
    for(int i = 0; i < vector_h; i++) { //for every row in the first matrix
        for(int j = 0; j < weight_w; j++) { //for every column in the second matrix
            product[i*weight_w+j] = 0.0;
            for(int k = 0; k < vector_w; k++) { //we compute the kth entry in row i of the INPUTS times the kth entry in column j of the WEIGHTS
                product[i*weight_w+j] += inputs[i*vector_w+k] * weights[k*weight_w+j];

            }
            product[i*weight_w+j] += bias[j];
        }
    }
}
__global__ void dotProductTransposeSegmented(float* inputs, float* weights, float * product, int vector_h, int vector_w, int weight_h, int weight_w, bool useHeight) {
    if(useHeight && vector_h == weight_h) {
        if(vector_w % (gridDim.x * gridDim.y) != 0 || weight_w % (blockDim.x * blockDim.y) != 0) {
            printf("BAD RESULT\n");
            return;
        }
        printf("Vector w %d weight w %d\n", vector_w, weight_w);
        int batch_size_x = vector_w / (gridDim.x);
        int batch_size_y = weight_w / (blockDim.x);
        int index_x = blockIdx.x;
        int index_y = threadIdx.x;
        printf("Batch size x %d Batch size y %d\n", batch_size_x, batch_size_y);
        printf("Index x %d, Index y %d\n", index_x, index_y);
        //index_x*batch_size_x indicates the starting row of the input matrix
        //index_x*batch_size_x*weight_x indicates the starting row of the product matrix
        //index_y*batch_size_y indicates the starting column of the weight matrix
        for(int i = 0; i < batch_size_x; i++) { //
            for(int j = 0; j < batch_size_y; j++) {
                printf("starting column in inputs: %d\nstarting column in weights: %d\n\n", index_x*batch_size_x+i, index_y*batch_size_y+j);
                product[(index_x*batch_size_x+i)*weight_w+(index_y*batch_size_y+j)] = 0.0f;
                for(int k = 0; k < vector_h; k++) {
                    product[(index_x*batch_size_x+i)*weight_w+(index_y*batch_size_y+j)] += inputs[index_x*batch_size_x+i+(vector_w*k)] * weights[index_y*batch_size_y+j+(weight_w*k)];
                }
                printf("%f\n", product[(index_x*batch_size_x+i)*weight_w+(index_y*batch_size_y+j)]);
            }
        }
    } else if(vector_w == weight_w) {
        if(vector_h % (gridDim.x * gridDim.y) != 0 || weight_h % (blockDim.x * blockDim.y) != 0) {
            printf("BAD RESULT\n");
            return;
        }
        int batch_size_x = vector_h / (gridDim.x*gridDim.y);
        int batch_size_y = weight_h / (blockDim.x*blockDim.y);
        // printf("Batch size x %d batch size y %d\n", batch_size_x, batch_size_y);
        int index_x = blockIdx.x*gridDim.y + blockIdx.y;
        int index_y = threadIdx.x*blockDim.y + threadIdx.y;
        // printf("Index x %d, Index y %d\n", index_x, index_y);
        for(int i = 0; i < batch_size_x; i++) {
            for(int j = 0; j < batch_size_y; j++) {
                printf("Index at x %d index at y %d, Start position in product array %d\n", index_x*batch_size_x+i, index_y*batch_size_y+j, (index_x*batch_size_x+i)*weight_h+index_y*batch_size_y+j);
                product[(index_x*batch_size_x+i)*weight_h+index_y*batch_size_y+j] = 0.0f;
                for(int k = 0; k < vector_w; k++) {
                    product[(index_x*batch_size_x+i)*weight_h+index_y*batch_size_y+j] += inputs[(index_x*batch_size_x+i)*vector_w+k] * weights[(index_y*batch_size_y+j)*weight_w+k];
                }
                printf("%f\n", product[(index_x*batch_size_x+i)*weight_h+index_y*batch_size_y+j]);
            }
        }
    }
}
__device__ void dotProductTranspose(float* inputs, float* weights, float * product, int vector_h, int vector_w, int weight_h, int weight_w) {
    //remember that we want the resulting matrix to be of shape [vector_h, weight_w]
    if(vector_h == weight_h) {
    for(int i = 0; i < vector_w; i++) {
        for(int j = 0; j < weight_w; j++) {
            product[i*weight_w+j] = 0.0;
            for(int k = 0; k < vector_h; k++) {
                product[i*weight_w+j] += inputs[k*vector_w+i] * weights[k*weight_w+j];
            }
        }
    }
    } else if(vector_w == weight_w) {
        for(int i = 0; i < vector_h; i++) {
            for(int j = 0; j < weight_h; j++) {
                product[i*weight_h+j] = 0.0;
                for(int k = 0; k < vector_w; k++) {
                    product[i*weight_h+j] += inputs[i*vector_w+k] * weights[j*weight_w+k];
                }
            }
        }
    } else {
        printf("INVALID DIMS FOR DOT PRODUCT\n");
    }
}

__global__ void matrixSubtract(float * matrix1, float *matrix2, int m1_h, int m1_w, int m2_h, int m2_w, float* outVec) {
    int index = blockIdx.x*blockDim.x + threadIdx.x;
    if((m1_h*m1_w) % (gridDim.x *blockDim.x) != 0) {
        printf("BAD INPUT\n");
    }
    //this can be literally be flattened out to a linear operation
    int batch_size = (m1_h*m1_w)/(gridDim.x *blockDim.x);
    for(int i = 0; i < batch_size; i++) {
        outVec[index*batch_size+i] = matrix1[index*batch_size+i] - matrix2[index*batch_size+i];
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
// __global__ void forward_pass(float* inputs, float* weights, float* outputs, float* product, float* gradients, int size, int n_features, int n_classes) {
//     int i = blockIdx.x*blockDim.x + threadIdx.x;
//     int batch = size / (blockDim.x * gridDim.x);
//     matrixSubtract(product+(i*n_classes*batch), outputs+(i*n_classes*batch), batch, n_classes, batch, n_classes, product+(i*n_classes*batch));
//     dotProductTranspose(inputs+(i*batch*n_features), product+(i*batch*n_classes), gradients+(i*n_features*n_classes), batch, n_features, batch, n_classes);
// }

// __global__ void forward_pass(LogisticRegression* model, float* inputs, float* outputs, float* product, int size, int nClasses) {
//     int i = blockIdx.x*blockDim.x + threadIdx.x;
//     int batch = size / (blockDim.x * gridDim.x);
//     matrixSubtract(product+(i*nClasses*batch), outputs+(i*nClasses*batch), batch, nClasses, batch, nClasses, product+(i*nClasses*batch));
//     dotProductTranspose(inputs+(i*batch*(model->nFeatures)), product+(i*batch*(model->nClasses)), ((*model).gradients)+(i*(model->nClasses)*(model->nFeatures)), batch, (model->nFeatures), batch, (model->nClasses));
// }

// __global__ void backward_pass(float* weights, float * gradients, int batch_size, float learning_rate, int n_features, int n_classes) {
//     int i = blockIdx.x*blockDim.x + threadIdx.x;
//     //BATCH_SIZE*n_classes length vector
//     int batch = n_features / (blockDim.x * gridDim.x);
//     matrixMultiplyByScalar(gradients+(i*batch*n_classes), batch, n_classes, learning_rate/(float) batch_size);
//     matrixSubtract(weights+(i*n_classes*batch), gradients+(i*n_classes*batch), batch, n_classes, batch, n_classes, weights+(i*n_classes*batch));
// }

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

///NEURAL NETWORK CODE

__global__ void predict(NeuralNetwork* model, float* inputs, float* product, int* offsets, int size) {
    int index = blockIdx.x*blockDim.x + threadIdx.x;
    float items = size / (float) (blockDim.x * gridDim.x);
    int batch = ceil(items);
    if (index * batch >= size) {
        return;
    }
    /*
    Each thread takes a chunk of the input data and feeds it all the way through the neural network, storing the intermediary results in product along the way.
    Just going to use softmax as the default activation as I'm more familiar with how that works anyways
    */
    float* input = inputs+(index*(model->layer_size[0])*batch);
    int batch_size = min(size-(index*batch), batch);
    // printf("batch size: %d\n", batch_size);
    float* out = (product+(index*(model->layer_size[1])*batch));
    dotProduct(input, model->weights[0], out, batch_size, model->layer_size[0], model->layer_size[0], model->layer_size[1], model->biases[0]);
    sigmoid(out, batch_size*model->layer_size[1]);
    for(int i = 1; i < model->nLayers; i++) {
        input = out;
        out = product+offsets[i]+(index*(model->layer_size[i+1])*batch);
        dotProduct(input, model->weights[i], out, batch_size, model->layer_size[i], model->layer_size[i], model->layer_size[i+1], model->biases[i]);
        softmax(out, batch_size, (model->layer_size[i+1]));
    }
}

// __global__ void backprop(NeuralNetwork* model, float* inputs, float* outputs, float* activations, float* deltas, int * offsets, int size, int nClasses) {
//     int index = blockIdx.x*blockDim.x + threadIdx.x;
//     float items = size / (float) (blockDim.x * gridDim.x);
//     int batch = ceil(items);
//     if (index * batch >= size) {
//         return;
//     }
//     /*
//     to do the forward pass, we need to take the dot product of the current activations and the previous activations. This is gonna take some effort, so maybe we should
//     */
//     /*Step 1: Subtract current predictions from the actual output (same step as in logistic regression)
//     But there's a caveat involved in where we actually store the results*/
//     int currentLayer = model->nLayers-1;
//     float* current = activations+offsets[currentLayer]+(index*nClasses*batch);
//     float* out = outputs+(index*nClasses*batch);
//     float* deltaPtr = deltas+offsets[currentLayer]+(index*nClasses*batch);
//     int batch_size = min(size-(index*batch), batch);
//     //compute deltas for the last layer;
//     matrixSubtract(current, out, batch_size, model->layer_size[currentLayer+1], batch_size, model->layer_size[currentLayer+1], deltaPtr); //[10X10 vector]
//     int bias_index = model->layer_size[currentLayer+1]*index;
//     // printf("Bias index %d\n", bias_index);
//     for(int j = 0; j < model->layer_size[currentLayer+1]; j++) {
//         model->grad_biases[currentLayer][bias_index+j] = 0.0;
//         for(int i = 0; i < batch_size; i++) {
//             // printf("Delta ptr %d %d %f\n", index, i, deltaPtr[i*model->layer_size[currentLayer+1]+j]);
//             model->grad_biases[currentLayer][bias_index+j] += deltaPtr[i*model->layer_size[currentLayer+1]+j];
//         }
//     }
//     // //main loop
//     for(int i = currentLayer; i > 0; i--) {
//         dotProductTranspose(deltaPtr, model->weights[i], deltas+offsets[i-1]+(index*model->layer_size[i]*batch), batch_size, model->layer_size[i+1], model->layer_size[i], model->layer_size[i+1]);
//         deltaPtr = deltas+offsets[i-1]+(index*model->layer_size[i]*batch);
//         current = activations+offsets[i-1]+((index*model->layer_size[i]*batch));

//         //mulitply delta by the derivative of the sigmoid function
//         sigmoidD(current, batch_size, model->layer_size[i], deltaPtr);

//         //compute gradients with respect to the biases
//         bias_index = model->layer_size[i]*index;
//         for(int j = 0; j < model->layer_size[i]; j++) {
//             model->grad_biases[i-1][bias_index+j] = 0.0;
//             for(int k = 0; k < batch_size; k++) {
//                 model->grad_biases[i-1][bias_index+j] += deltaPtr[k*(model->layer_size[i])+j];
//             }
//         }
//     }
//     currentLayer = model->nLayers-1;
//     float * activationPtr; 
//     for(int i = currentLayer; i > 0; i--) {
//         deltaPtr = deltas+offsets[i]+(index*model->layer_size[i+1]*batch);
//         activationPtr = activations+offsets[i-1]+((index*model->layer_size[i]*batch)); 
//         int gradientIndex = (index*model->layer_size[i+1]*model->layer_size[i]);
//         dotProductTranspose(activationPtr, deltaPtr, model->gradients[i]+gradientIndex, batch_size, model->layer_size[i], batch_size, model->layer_size[i+1]);
//     }
//     deltaPtr = deltas+(index*model->layer_size[1]*batch);
//     int gradientIndex = (index*model->layer_size[0]*model->layer_size[1]);
//     dotProductTranspose(inputs+(model->layer_size[0]*batch*index), deltaPtr, model->gradients[0]+gradientIndex, batch_size, model->layer_size[0], batch_size, model->layer_size[1]);
// }

__global__ void auditDeltas(NeuralNetwork* model, float * deltas, int* offsets, int batches, int batch_size) {
    float* deltaPtr;
    for(int i = 1; i <= model->nLayers; i++) {
        printf("Deltas for layer %d %d\n", i, offsets[i-1]);
        deltaPtr = deltas+offsets[i-1];
        for(int j = 0; j < batch_size; j++) {
            for(int k = 0; k < model->layer_size[i]-1; k++) {
                printf("%f\t", deltaPtr[j*model->layer_size[i]+k]);
            }
            printf("%f\n", deltaPtr[(j+1)*(model->layer_size[i])-1]);
        }
    }
}
__global__ void auditGradients(NeuralNetwork* model) {
    printf("Audit\n");
    for(int i = 0; i < model->nLayers; i++) {
        printf("Gradients for weights %d\n", i);
        for(int j = 0; j < model->layer_size[i]; j++) {
            for(int k = 0; k < model->layer_size[i+1]-1; k++) {
                printf("%f\t", model->gradients[i][(j*model->layer_size[i+1])+k]);
            }
            printf("%f\n", model->gradients[i][(j+1)*model->layer_size[i+1]-1]);
        }

        // printf("Gradients for biases %d\n", i);
        // for(int j = 0; j < model->layer_size[i+1]-1; j++) {
        //     printf("%f\t", model->grad_biases[i][j]);
        // }
        // printf("%f\n", model->grad_biases[i][model->layer_size[i+1]-1]);
    }
    printf("Success\n");
}
__global__ void auditWeights(NeuralNetwork* model) {
    for(int i = 0; i < model->nLayers; i++) {
        printf("Weights at layer %d\n", i);
        for(int j = 0; j < model->layer_size[i]; j++) {
            for(int k = 0; k < model->layer_size[i+1]-1; k++) {
                printf("%f\t", model->weights[i][j*model->layer_size[i+1]+k]);
            }
            printf("%f\n", model->weights[i][(j*model->layer_size[i+1])+model->layer_size[i+1]-1]);
        }
    }
}

__global__ void test_func(float* mat, float* weights, float* prod, int vector_h, int vector_w) {
    printf("Successful significant\n");
    printf("%f %f\n", mat[0], mat[1]);
}

__global__ void ringReduce(NeuralNetwork* model, const int total_steps) {
    int index = blockIdx.x*blockDim.x + threadIdx.x;
    for(int i = 0; i < model->nLayers; i++) {

        //reduce gradients[i]
        int step_size = (model->layer_size[i] * model->layer_size[i+1]);
        float step = (step_size) /(float) (blockDim.x * gridDim.x);
        int batch = ceil(step);
        // printf("Batch size for index %d of gradients %d: %d\n", index, i, batch);
        int start = index*batch;
        if(start >= step_size) {
            return;
        }
        for(int j = 1; j < total_steps; j++) {
            for(int k = start; k < min(start+batch, step_size); k++) {
                model->gradients[i][k] += model->gradients[i][k+(j*step_size)];
            }
        }

        //reduce biases[i]
        step_size = model->layer_size[i+1];
        step = step_size / (float) (blockDim.x * gridDim.x);
        batch = ceil(step);
        start = index*batch;
        if(start >= step_size) {
            return;
        }
        for(int j = 1; j < total_steps; j++) {
            for(int k = start; k < min(start+batch, step_size); k++) {
                // printf("Entry %d %d\n", j, k);
                model->grad_biases[i][k] += model->grad_biases[i][k+(j*step_size)];
            }
        }
    }
}

__global__ void backward_pass(NeuralNetwork* model, int batch_size, float learning_rate) {
    int index = blockIdx.x*blockDim.x + threadIdx.x;
    //BATCH_SIZE*n_classes length vector
    for(int k = 0; k < model->nLayers; k++) {
        // printf("Layer %d\n", k);
        int size = (model->layer_size[k] * model->layer_size[k+1]);
        float step = size / (float) (blockDim.x * gridDim.x);
        int batch = ceil(step);
        int start = index*batch;
        // printf("Gradients for")
        // printf("Starting index %d %d %d\n", k, index, start);
        if(start >= size) {
            return;
        }
        for(int i = start; i < min(start+batch, size); i++) {
            (*model).gradients[k][i] *= (learning_rate) / (float) batch_size;
            (*model).weights[k][i] -=  (*model).gradients[k][i];
            // printf("WEIGHT AT %d %d: %f\n", k, i, (*model).weights[k][i]);
        }

        size = model->layer_size[k+1];
        step = size / (float) (blockDim.x * gridDim.x);
        batch = ceil(step);
        start = index*batch;
        if(start >= size) {
            return;
        }
        for(int i = start; i < min(start+batch, size); i++) {
            // printf("Entry %d %d\n", j, k);
            model->grad_biases[k][i] *= (learning_rate) / (float) batch_size;
            model->biases[k][i] -= model->grad_biases[k][i];
        }
    }
    // printf("Finish backward\n");
}

__global__ void dotProductSegmented(float* inputs, float* weights, float * product, int vector_h, int vector_w, int weight_h, int weight_w) {
    printf("This is called\n");
    int index = blockIdx.x*blockDim.x + threadIdx.x;
    int index_x = blockIdx.z*blockDim.y + blockIdx.y;
    int index_y = threadIdx.z*gridDim.y + threadIdx.y;
    if(vector_h % (gridDim.y*gridDim.z) != 0 || weight_w % (blockDim.y*blockDim.z)) {
        printf("BAD ARGUMENTS\n");
        return;
    }
    int size_x = vector_h / (gridDim.y*gridDim.z);
    int size_y = weight_w / (blockDim.y*blockDim.z);
    printf("Size_x: %d\nSize_y: %d\n", size_x, size_y);
    float* out = product+(size_x*index_x*weight_w)+(size_y*index_y); 
    float* input = inputs+((size_x*index_x*vector_w));
    float* weight = weights+(size_y*index_y);
    for(int i = 0; i < size_x; i++) { //for every row in the first matrix
        for(int j = 0; j < size_y; j++) { //for every column in the second matrix
            out[i*weight_w+j] = 0.0;
            for(int k = 0; k < vector_w; k++) { //we compute the kth entry in row i of the INPUTS times the kth entry in column j of the WEIGHTS
                out[i*weight_w+j] += input[i*vector_w+k] * weight[k*weight_w+j];
                // printf("This %d %d %f %f\n", i, j, inputs[i*vector_w+k], weights[k*weight_w+j]);
            }
        }
    }
}

__global__ void dotProductSegmented(float* inputs, float* weights, float * product, int vector_h, int vector_w, int weight_h, int weight_w, float* bias) {
    printf("Successful call\n");
    printf("Input start %f %f\n", inputs[0], product[0]);
    int index = blockIdx.x*blockDim.x + threadIdx.x;
    //we subdivide vector_h in "mini-batches" of size batch_size
    //we need to check that (blockDim.x * gridDim.x) is divisible by vector_h
    if(vector_h % (blockDim.x * gridDim.x) != 0) {
        printf("BAD ARGUMENT for global batch size\n");
        return;
    }
    int batch_size = vector_h / (blockDim.x * gridDim.x);
    int index_x = blockIdx.z*blockDim.y + blockIdx.y;
    int index_y = threadIdx.z*gridDim.y + threadIdx.y;
    //and we subdivide batch_size further by dividing by gridDim.y*gridDim.z
    if((batch_size % (gridDim.y*gridDim.z) != 0) || (weight_w % (blockDim.y*blockDim.z) != 0)) {
        printf("BAD ARGUMENTS\n");
        printf("Size_x %d %d\nSize_y %d %d\n", vector_h, gridDim.y*gridDim.z, weight_w, blockDim.y*blockDim.z);
        return;
    }
    if((batch_size < gridDim.y*gridDim.z) || (weight_w < blockDim.y*blockDim.z)) {
        printf("BAD ARGUMENTS\n");
        printf("Size_x %d %d\nSize_y %d %d\n", batch_size, gridDim.y*gridDim.z, weight_w, blockDim.y*blockDim.z);
        return; 
    }
    int size_x = batch_size / (gridDim.y*gridDim.z);
    int size_y = weight_w / (blockDim.y*blockDim.z);
    // printf("Start for %d %d %d\n", index, index*batch_size*vector_w, index*batch_size*weight_w);
    float* out = product+(index*batch_size*weight_w)+(size_x*index_x*weight_w)+(size_y*index_y); 
    float* input = inputs+(index*batch_size*vector_w)+((size_x*index_x*vector_w));
    float* weight = weights+(size_y*index_y);
    for(int i = 0; i < size_x; i++) { //for every row in the first matrix
        for(int j = 0; j < size_y; j++) { //for every column in the second matrix
            out[i*weight_w+j] = 0.0;
            for(int k = 0; k < vector_w; k++) { //we compute the kth entry in row i of the INPUTS times the kth entry in column j of the WEIGHTS
                out[i*weight_w+j] += input[i*vector_w+k] * weight[k*weight_w+j];
            }
            out[i*weight_w+j] += bias[j+(index_y*size_y)];
        }
    }
}

__global__ void sigmoidSegmented(float* inputs, int inputSize) {
    if(inputSize % (blockDim.x * gridDim.x) != 0) {
        printf("BAD ARGUMENT\n");
        return;
    }
    int blockSize = inputSize / (blockDim.x * gridDim.x);
    int index = blockIdx.x*blockDim.x + threadIdx.x;
    sigmoid(inputs+(blockSize*index), blockSize);

}

__global__ void softmaxSegmented(float* product, int product_height, int product_width) {
    if(product_height % (blockDim.x * gridDim.x) != 0) {
        printf("BAD ARGUMENT\n");
        return;
    }
    int blockSize = product_height / (blockDim.x * gridDim.x);
    int index = blockIdx.x*blockDim.x + threadIdx.x;
    softmax(product+(blockSize*index), blockSize, product_width);
}

