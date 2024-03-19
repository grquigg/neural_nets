#include "../include/nn.h"
#include "../include/utils.h"
#include "../include/lin_alg.h"
#include <chrono> 
#include <cublas_v2.h>

NeuralNetwork * copyModelToGPU(NeuralNetwork *model, int nWorkers, int nThreadsPerWorker) {
    NeuralNetwork* d_model;
    int * nLayers;
    float **d_weights;
    float **d_weights_t;
    float **d_biases;
    float **d_gradients;
    float **d_grad_biases;
    //allocate all of the memory that we need to CUDA
    cudaMalloc(&d_model, sizeof(NeuralNetwork));
    cudaMalloc(&nLayers, (model->nLayers+1)*sizeof(int));
    cudaMemcpy(nLayers, model->layer_size, (model->nLayers+1)*sizeof(int), cudaMemcpyHostToDevice);
    // // cudaMalloc(&d_weights, (model->nLayers)*sizeof(float*));
    // cudaMalloc(&d_biases, (model->nLayers)*sizeof(float*));
    float **temp_weights = new float*[model->nLayers];
    float **temp_biases = new float*[model->nLayers];
    float **temp_gradients = new float*[model->nLayers];
    float **temp_grad_biases = new float*[model->nLayers];
    for(int i = 1; i < model->nLayers+1; i++) {
        cudaMalloc(&temp_weights[i-1], model->layer_size[i-1]*model->layer_size[i]*sizeof(float));
        cudaMemcpy(temp_weights[i-1], model->weights[i-1], model->layer_size[i-1]*model->layer_size[i]*sizeof(float), cudaMemcpyHostToDevice);
        cudaMalloc(&temp_biases[i-1], model->layer_size[i]*sizeof(float));
        cudaMemcpy(temp_biases[i-1], model->biases[i-1], model->layer_size[i]*sizeof(float), cudaMemcpyHostToDevice);
        cudaMalloc(&temp_gradients[i-1], nThreadsPerWorker*nWorkers*model->layer_size[i-1]*model->layer_size[i]*sizeof(float));
        // cudaMemcpy(temp_gradients[i-1], model->gradients[i-1], nThreadsPerWorker*nWorkers*model->layer_size[i-1]*model->layer_size[i]*sizeof(float), cudaMemcpyHostToDevice);
        cudaMalloc(&temp_grad_biases[i-1],  nThreadsPerWorker*nWorkers*model->layer_size[i]*sizeof(float));
    }
    cudaMalloc(&d_gradients, (model->nLayers)*sizeof(float*));
    cudaMemcpy(d_gradients, temp_gradients, (model->nLayers)*sizeof(float*), cudaMemcpyHostToDevice);
    cudaMalloc(&d_grad_biases, (model->nLayers)*sizeof(float*));
    cudaMemcpy(d_grad_biases, temp_grad_biases, (model->nLayers)*sizeof(float*), cudaMemcpyHostToDevice);
    cudaMalloc(&d_biases, (model->nLayers)*sizeof(float*));
    cudaMemcpy(d_biases, temp_biases, (model->nLayers)*sizeof(float*), cudaMemcpyHostToDevice);
    cudaMalloc(&d_weights, (model->nLayers)*sizeof(float*));
    cudaMemcpy(d_weights, temp_weights, (model->nLayers)*sizeof(float*), cudaMemcpyHostToDevice);
    NeuralNetwork temp = *model;
    temp.nClasses = model->nClasses;
    temp.nLayers = model->nLayers;
    temp.layer_size = nLayers;
    temp.weights = d_weights;
    temp.gradients = d_gradients;
    temp.biases = d_biases;
    temp.grad_biases = d_grad_biases;
    temp.lambda = model->lambda;
    cudaMemcpy(d_model, &temp, sizeof(NeuralNetwork), cudaMemcpyHostToDevice);
    return d_model;
}

void train(NeuralNetwork *model, float* train_input, std::vector<std::vector<int>>& train_labels, float* test_input, std::vector<std::vector<int>>& test_labels, 
int nEpochs, int batch_size, int total_size, int test_size, float learning_rate, int nWorkers, int nThreadsPerWorker) {
    NeuralNetwork *d_model = copyModelToGPU(model, nWorkers, nThreadsPerWorker);
    //copy train data

    //initialize cublas logic
    cublasHandle_t handle;
    cublasCreate(&handle);
    cudaError_t error;
    float *d_inputs;
    //copy weights
    error = cudaMalloc(&d_inputs, total_size*(model->layer_size[0])*sizeof(float));
    if(error != cudaSuccess) {
        std::cout << "Problem with copying" << std::endl;
    }
    error = cudaMemcpy(d_inputs, train_input, total_size*(model->layer_size[0])*sizeof(float), cudaMemcpyHostToDevice);
    if(error != cudaSuccess) {
        std::cout << "Problem" << std::endl;
    }
    //copy test data
    float *d_test_inputs;
    cudaMalloc(&d_test_inputs, test_size*(model->layer_size[0])*sizeof(float));
    cudaMemcpy(d_test_inputs, test_input, test_size*(model->layer_size[0])*sizeof(float), cudaMemcpyHostToDevice);

    //convert labels to one hot encoding
    float * one_hot = (float *)malloc(sizeof(float) * total_size * model->nClasses);
    for (int i = 0; i < total_size; i++) {
        for(int j = 0; j < model->nClasses; j++) {
            one_hot[i*model->nClasses+j] = 0;
        }
        one_hot[i*(model->nClasses)+train_labels[i][0]] = 1.0;
    }
    //pass labels to GPU
    float *d_outputs = transferMatrixToDevice(one_hot, total_size, model->nClasses);

    //initialize array for storing intermediary activation functions on GPU
    /*the super nice thing about the parallelized computation of neural networks is 
    ALL of the activation functions take the form of (BATCH_SIZE, layer_size)
    Which means we can likely have all of the activations stored via one pointer and only
    have to allocate the memory ONCE. However, since I have absolutely no idea what I'm doing,
    I'm gonna stay away from that for now.

    Since double pointers don't want to cooperate for some reason, and since it doesn't make sense
    for this huge block of memory to be allocated several different times randomly in memory, we allocate a single block
    of memory as well as an integer array to keep track of the offsets of each "block" in memory.
    */
    int activations_size = 0;
    int * offsets = new int[model->nLayers];
    for(int i = 1; i <= model->nLayers; i++) {
        offsets[i-1] = (batch_size * activations_size);
        // printf("Offset at %d: %d\n", i-1, offsets[i-1]);
        activations_size += model->layer_size[i];
    }
    float * d_activations = new float[batch_size*activations_size];
    float * activations = new float[batch_size*activations_size];
    //device pointers
    int * d_offsets;
    cudaMalloc(&d_activations, activations_size*batch_size*sizeof(float));
    cudaMalloc(&d_offsets, model->nLayers*sizeof(int));
    for(int i = 0; i < activations_size*batch_size; i++) {
        activations[i] = 1;
    }
    cudaMemcpy(d_activations, activations, activations_size*batch_size*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_offsets, offsets, model->nLayers*sizeof(int), cudaMemcpyHostToDevice);

    //deltas
    float * d_deltas = new float[batch_size*activations_size];
    cudaMalloc(&d_deltas, activations_size*batch_size*sizeof(float));
    cudaMemcpy(d_activations, activations, activations_size*batch_size*sizeof(float), cudaMemcpyHostToDevice);
    // float * d_product = transferMatrixToDevice(activations, batch_size, activations_size);
    // //initialize array for storing predictions of test set on host
    float* predictions = (float*)malloc(activations_size*batch_size*sizeof(float));
    float * test_predictions = (float*)malloc(test_size*model->nClasses*sizeof(float));
    float * d_test_product = transferMatrixToDevice(test_predictions, test_size, model->nClasses);
    //define metrics
    int correct = 0;
    double logLoss = 0.0;
    float accuracy = 0.0;
    int totalEpochs = 0;
    std::cout << "START TRAIN" << std::endl;
    auto startTrain = std::chrono::system_clock::now();
    for(int i = 0; i < nEpochs; i++) {
        correct = 0;
        logLoss = 0;
        accuracy = 0.0;
        for(int j = 0; j < total_size; j+=batch_size) {
            //pass inputs through the network
            auto startForward = std::chrono::system_clock::now();
            printf("Start predictions\n");
            predict(d_model, d_inputs+(j*model->layer_size[0]), d_activations, d_offsets, batch_size, handle);
            cudaDeviceSynchronize();
            // auto endForward = std::chrono::system_clock::now();
            // std::chrono::duration<double> elapsed_forward = endForward - startForward;
            // std::cout << "Finished forward pass in " << elapsed_forward.count() << " seconds" << std::endl;
            error = cudaMemcpy(predictions, d_activations, activations_size*batch_size*sizeof(float), cudaMemcpyDeviceToHost);
            correct += getAccuracy(predictions+offsets[1], train_labels, batch_size, model->nClasses, j);
            logLoss += crossEntropyLoss(predictions+offsets[1], train_labels, batch_size, model->nClasses, j);
            // printf("Accuracy: %f%%\n", correct / (float) (batch_size)* 100);
            printf("Log loss %f\n", (logLoss * batch_size)/(float)(j+1));
            // //compute gradients in forward_pass
            // auto startBackward = std::chrono::system_clock::now();
            // backprop<<<nWorkers, nThreadsPerWorker>>>(d_model, d_inputs+(j*(model->layer_size[0])), d_outputs+(j*(model->nClasses)), d_activations, d_deltas, d_offsets, batch_size, model->nClasses);
            cudaDeviceSynchronize();
            // ringReduce<<<nWorkers, nThreadsPerWorker>>>(d_model, nWorkers*nThreadsPerWorker);
            cudaDeviceSynchronize();
            // auto endReduce = std::chrono::system_clock::now();
            // std::chrono::duration<double> elapsed_reduce = endReduce - startReduce;
            // std::cout << "Finished ring reduce in " << elapsed_reduce.count() << " seconds" << std::endl;
            // auditDeltas<<<1,1>>>(d_model, d_deltas, d_offsets, nWorkers*nThreadsPerWorker, batch_size);
            // cudaDeviceSynchronize();
            // auditGradients<<<1,1>>>(d_model);
            // cudaDeviceSynchronize();
            auto startUpdate = std::chrono::system_clock::now();
            // backward_pass<<<nWorkers, nThreadsPerWorker>>>(d_model, batch_size, learning_rate);
            cudaDeviceSynchronize();
            // auditWeights<<<1,1>>>(d_model);
            // cudaDeviceSynchronize();
            // auto endUpdate = std::chrono::system_clock::now();
            // std::chrono::duration<double> elapsed_update = endUpdate - startUpdate;
            // std::cout << "Finished weight update in " << elapsed_update.count() << " seconds" << std::endl;
            totalEpochs++;
        }
        accuracy = correct / (float) total_size;
        printf("End of epoch %d\n", i+1);
        printf("Accuracy: %f%%\n", accuracy*100);
    }
    auto endTrain = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed_forward = endTrain - startTrain;
    std::cout << "Finished forward pass in " << elapsed_forward.count() << " seconds" << std::endl;
    for(int i = 1; i < model->nLayers+1; i++) {
        cudaFree(d_model->weights[i-1]);
        cudaFree(d_model->biases[i-1]);
        cudaFree(d_model->gradients[i-1]);
        // cudaMemcpy(temp_gradients[i-1], model->gradients[i-1], nThreadsPerWorker*nWorkers*model->layer_size[i-1]*model->layer_size[i]*sizeof(float), cudaMemcpyHostToDevice);
        cudaFree(d_model->grad_biases[i-1]);
    }
    cudaFree(d_model->layer_size);
    cudaFree(d_model);
    cudaFree(d_inputs);
    cudaFree(d_test_inputs);
    cudaFree(d_outputs);
    cudaFree(d_activations);
    cudaFree(d_test_product);
    cudaFree(d_deltas);
    
}

void predict(NeuralNetwork* model, float* inputs, float* product, int* offsets, int size, cublasHandle_t handle) {
    int batch = size;
    int index = 0;
    // if (index * batch >= size) {
    //     return;
    // }
    /*
    Each thread takes a chunk of the input data and feeds it all the way through the neural network, storing the intermediary results in product along the way.
    Just going to use softmax as the default activation as I'm more familiar with how that works anyways
    */
    float alpha = 1.0f;
    float beta = 0.0f;
    printf("Okay\n");
    //args are as follows: handle, transa, transb, #rows in input, #cols in model->weights[0], #rows in model->weights[0], &alpha, inputs, size, model
    cublasStatus_t error = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, size, model->layer_size[1], model->layer_size[0], &alpha, inputs, size, model->weights[0], model->layer_size[0], &beta, product, size);
    printf("ERROR: %d\n", error);
    cudaDeviceSynchronize();
    printf("Success\n");
    // sigmoid(out, batch_size*model->layer_size[1]);
    // for(int i = 1; i < model->nLayers; i++) {
    //     input = out;
    //     out = product+offsets[i]+(index*(model->layer_size[i+1])*batch);
    //     dotProduct(input, model->weights[i], out, batch_size, model->layer_size[i], model->layer_size[i], model->layer_size[i+1], model->biases[i]);
    //     softmax(out, batch_size, (model->layer_size[i+1]));
    // }
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
