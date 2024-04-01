#include "../include/utils.h"
#include "../include/models.h"
#include "../include/lin_alg.h"
#include <chrono> 
#include <iostream>
#include <cublas_v2.h>
#include <memory>

void free2DArrayFromDevice(float ** array, int * array_size) {

}

std::shared_ptr<float> transferMatrixToDevice(float *matrix, int height, int width) {
    float* deviceMatrix = nullptr;
    cudaMalloc(&deviceMatrix, height*width*sizeof(float));
    for(int i = 0; i < height; i++) {
        cudaMemcpy(deviceMatrix+(i*width), matrix+(i*width), sizeof(float)*width, cudaMemcpyHostToDevice);
    }
    std::shared_ptr<float> return_mat = std::shared_ptr<float>(deviceMatrix, CudaDeallocator());
    return return_mat;
}

std::shared_ptr<int> transferMatrixToDevice(int * matrix, int height, int width) {
    int* deviceMatrix = nullptr;
    cudaMalloc(&deviceMatrix, height*width*sizeof(int));
    for(int i = 0; i < height; i++) {
        cudaMemcpy(deviceMatrix+(i*width), matrix+(i*width), sizeof(int)*width, cudaMemcpyHostToDevice);
    }
    std::shared_ptr<int> return_mat = std::shared_ptr<int>(deviceMatrix, CudaDeallocator());
    return return_mat;
}

NeuralNetwork::NeuralNetwork(int nLayers, int * layer_size) {
    this->nLayers = nLayers;
    this->layer_size = layer_size;
}

NeuralNetwork::~NeuralNetwork() {
    if(this->on_device) {
        for(int i = 0; i < nLayers; i++) {
            cudaFree(this->weights[i]);
            cudaFree(this->biases[i]);
        }
        cudaFree(this->layer_size);
    }
}

NeuralNetwork::NeuralNetwork() {}

NeuralNetwork::NeuralNetwork(int nLayers, int * layer_size, float** weights, float ** biases, float lambda) {
    this->nLayers = nLayers;
    this->layer_size = layer_size;
    this->weights = weights;
    this->biases = biases;
    this->lambda = lambda;
}

void NeuralNetwork::setupDeltas(int batch_size) {
    this->deltas = new float*[this->nLayers];
    for(int i = 0; i < this->nLayers; i++) {
        cudaMalloc(&this->deltas[i], this->layer_size[i+1]*batch_size*sizeof(float));
    }
}

void NeuralNetwork::setupGPU(int nThreads) {
    this->d_weights = new float*[this->nLayers];
    this->d_biases = new float*[this->nLayers];
    this->gradients = new float*[this->nLayers];
    this->grad_biases = new float*[this->nLayers];
    std::cout << "Starting to set" << std::endl;
    for(int i = 1; i < this->nLayers+1; i++) {
        cudaMalloc(&this->d_weights[i-1], this->layer_size[i-1]*this->layer_size[i]*sizeof(float));
        cudaMemcpy(this->d_weights[i-1], this->weights[i-1], this->layer_size[i-1]*this->layer_size[i]*sizeof(float), cudaMemcpyHostToDevice);
        cudaMalloc(&this->d_biases[i-1], this->layer_size[i]*sizeof(float));
        cudaMemcpy(this->d_biases[i-1], this->biases[i-1], this->layer_size[i]*sizeof(float), cudaMemcpyHostToDevice);
        cudaMalloc(&this->gradients[i-1], nThreads*this->layer_size[i-1]*this->layer_size[i]*sizeof(float));
        cudaMalloc(&this->grad_biases[i-1], nThreads*this->layer_size[i]*sizeof(float));
    }
    // cudaMalloc(&this->gradients, (this->nLayers)*sizeof(float*));
    // cudaMemcpy(this->gradients, temp_gradients, (this->nLayers)*sizeof(float*), cudaMemcpyHostToDevice);
    // cudaMalloc(&this->grad_biases, (this->nLayers)*sizeof(float*));
    // cudaMemcpy(this->grad_biases, temp_grad_biases, (this->nLayers)*sizeof(float*), cudaMemcpyHostToDevice);
    // cudaMalloc(&this->d_biases, (this->nLayers)*sizeof(float*));
    // cudaMemcpy(this->d_biases, temp_biases, (this->nLayers)*sizeof(float*), cudaMemcpyHostToDevice);
    // cudaMalloc(&this->d_weights, (this->nLayers)*sizeof(float*));
    // cudaMemcpy(this->d_weights, temp_weights, (this->nLayers)*sizeof(float*), cudaMemcpyHostToDevice);
    std::cout << "Successful pass" << std::endl;
}
void NeuralNetwork::backprop(int batch_size, std::shared_ptr<float> inputs, std::shared_ptr<float> outputs) {
//       matrixSubtract<<<this->layer_size[this->nLayers],batch_size>>>(this->activations+this->offsets[this->nLayers-1], d_y.get(), batch_size, this->layer_size[this->nLayers], batch_size, this->layer_size[this->nLayers], d_deltas);
//   float* d_deltas0;
//   cudaMalloc(&d_deltas0, batch_size*this->layer_size[this->nLayers-1]*sizeof(float));
//   std::cout << batch_size << " " << this->layer_size[this->nLayers-1] << std::endl;
//   //there's ambiguity in the dotProductTransposeSegmented
//   dotProductTransposeSegmented<<<batch_size,this->nLayers-1>>>(d_deltas, this->d_weights[this->nLayers-1], d_deltas0, batch_size, this->layer_size[this->nLayers], this->layer_size[this->nLayers-1], this->layer_size[this->nLayers], false);
    matrixSubtract<<<batch_size,this->layer_size[this->nLayers]>>>(this->activations+this->offsets[this->nLayers-1], outputs.get(), batch_size, this->layer_size[this->nLayers], batch_size, this->layer_size[this->nLayers], this->deltas[this->nLayers-1]);
    cudaDeviceSynchronize();
    for(int i = this->nLayers-1; i > 0; i--) {
        dotProductTransposeSegmented<<<batch_size, this->layer_size[i]>>>(this->deltas[i], this->d_weights[i], this->deltas[i-1], batch_size, this->layer_size[i+1], this->layer_size[i], this->layer_size[i+1], false);
        cudaDeviceSynchronize();
        sigmoidD<<<batch_size, this->layer_size[i]>>>(this->activations+this->offsets[i-1], batch_size, this->layer_size[i], this->deltas[i-1]);
        cudaDeviceSynchronize();
    }
    // for(int i = this->nLayers - 1; i > 0; i--) {
    //     std::cout << "Layer " << i << std::endl;
    //     dotProductTransposeSegmented<<<batch_size, this->layer_size[i-1]>>>(this->deltas[i], this->d_weights[i], this->deltas[i-1], batch_size, this->layer_size[i], this->layer_size[i-1], this->layer_size[i], false);
    //     cudaDeviceSynchronize();
    //     // sigmoidD<<<batch_size, this->layer_size[i-1]>>>(this->activations+this->offsets[i-1], batch_size, this->layer_size[i], this->deltas[i-1]);
    //     // cudaDeviceSynchronize();
    // }
    std::cout << "Made it to the end" << std::endl;
}

std::shared_ptr<float> NeuralNetwork::forward_pass(std::shared_ptr<float> d_input, int total_size, int batch_size, int nWorkers, int nThreadsPerWorkers) {
    dim3 nBlocks(nWorkers, 1, 1);
    dim3 nThreads(nThreadsPerWorkers, 1, 1);
    int activations_size = 0;
    this->offsets = new int[this->nLayers];
    for(int i = 1; i <= this->nLayers; i++) {
        this->offsets[i-1] = (batch_size * activations_size);
        activations_size += this->layer_size[i];
    }
    std::shared_ptr<int> d_offsets = transferMatrixToDevice(this->offsets, this->nLayers, 1);
    float * d_activations;
    cudaMalloc(&d_activations, batch_size*activations_size*sizeof(float));
    // for(int i = 0; i < total_size; i+=batch_size) {
    dotProductSegmented<<<nBlocks, nThreads>>>(d_input.get(), this->d_weights[0], d_activations, batch_size, this->layer_size[0], this->layer_size[0], this->layer_size[1], this->d_biases[0]);
    cudaDeviceSynchronize();
    sigmoidSegmented<<<nWorkers, nThreadsPerWorkers>>>(d_activations, batch_size*this->layer_size[1]);
    cudaDeviceSynchronize();
    int j = 1;
    for(j = 1; j < this->nLayers-1; j++) {
        nThreads.y = this->layer_size[j+1];
        dotProductSegmented<<<nBlocks, nThreads>>>(d_activations+this->offsets[j-1], this->d_weights[j], d_activations+this->offsets[j], batch_size, this->layer_size[j], this->layer_size[j], this->layer_size[j+1], this->d_biases[j]);
        cudaDeviceSynchronize();
        sigmoidSegmented<<<nWorkers, nThreadsPerWorkers>>>(d_activations+this->offsets[j], batch_size*this->layer_size[j+1]);
        cudaDeviceSynchronize();
    }
    nThreads.y = this->layer_size[j+1];
    dotProductSegmented<<<nBlocks, nThreads>>>(d_activations+this->offsets[j-1], this->d_weights[j], d_activations+this->offsets[j], batch_size, this->layer_size[j], this->layer_size[j], this->layer_size[j+1], this->d_biases[j]);
    cudaDeviceSynchronize();
    softmaxSegmented<<<nWorkers, nThreadsPerWorkers>>>(d_activations+(this->offsets[j]), batch_size, this->layer_size[j+1]);
    cudaDeviceSynchronize();
    float * activations = new float[activations_size*batch_size];
    cudaMemcpy(activations, d_activations, activations_size*batch_size*sizeof(float), cudaMemcpyDeviceToHost);
    this->activations = d_activations;
    return std::shared_ptr<float>(activations);
}

// void train(NeuralNetwork *model, float* train_input, std::vector<std::vector<int>>& train_labels, float* test_input, std::vector<std::vector<int>>& test_labels, 
// int nEpochs, int batch_size, int total_size, int test_size, float learning_rate, int nWorkers, int nThreadsPerWorker, bool useMultiThreaded) {
//     //copy train data
//     float *d_inputs;
//     //copy weights
//     cudaError_t error;
//     error = cudaMalloc(&d_inputs, total_size*(model->layer_size[0])*sizeof(float));
//     if(error != cudaSuccess) {
//         std::cout << "Problem with copying" << std::endl;
//     }
//     error = cudaMemcpy(d_inputs, train_input, total_size*(model->layer_size[0])*sizeof(float), cudaMemcpyHostToDevice);
//     if(error != cudaSuccess) {
//         std::cout << "Problem" << std::endl;
//     }
//     //copy test data
//     float *d_test_inputs;
//     cudaMalloc(&d_test_inputs, test_size*(model->layer_size[0])*sizeof(float));
//     cudaMemcpy(d_test_inputs, test_input, test_size*(model->layer_size[0])*sizeof(float), cudaMemcpyHostToDevice);

//     //convert labels to one hot encoding
//     float * one_hot = (float *)malloc(sizeof(float) * total_size * model->nClasses);
//     for (int i = 0; i < total_size; i++) {
//         for(int j = 0; j < model->nClasses; j++) {
//             one_hot[i*model->nClasses+j] = 0;
//         }
//         one_hot[i*(model->nClasses)+train_labels[i][0]] = 1.0;
//     }
//     //pass labels to GPU
//     std::shared_ptr<float> d_outputs = transferMatrixToDevice(one_hot, total_size, model->nClasses);

//     //initialize array for storing intermediary activation functions on GPU
//     /*the super nice thing about the parallelized computation of neural networks is 
//     ALL of the activation functions take the form of (BATCH_SIZE, layer_size)
//     Which means we can likely have all of the activations stored via one pointer and only
//     have to allocate the memory ONCE. However, since I have absolutely no idea what I'm doing,
//     I'm gonna stay away from that for now.

//     Since double pointers don't want to cooperate for some reason, and since it doesn't make sense
//     for this huge block of memory to be allocated several different times randomly in memory, we allocate a single block
//     of memory as well as an integer array to keep track of the offsets of each "block" in memory.
//     */
//     int activations_size = 0;
//     int * offsets = new int[model->nLayers];
//     for(int i = 1; i <= model->nLayers; i++) {
//         offsets[i-1] = (batch_size * activations_size);
//         // printf("Offset at %d: %d\n", i-1, offsets[i-1]);
//         activations_size += model->layer_size[i];
//     }
//     float * d_activations = new float[batch_size*activations_size];
//     float * activations = new float[batch_size*activations_size];
//     //device pointers
//     int * d_offsets;
//     cudaMalloc(&d_activations, activations_size*batch_size*sizeof(float));
//     cudaMalloc(&d_offsets, model->nLayers*sizeof(int));
//     for(int i = 0; i < activations_size*batch_size; i++) {
//         activations[i] = 1;
//     }
//     cudaMemcpy(d_activations, activations, activations_size*batch_size*sizeof(float), cudaMemcpyHostToDevice);
//     cudaMemcpy(d_offsets, offsets, model->nLayers*sizeof(int), cudaMemcpyHostToDevice);

//     //deltas
//     float * d_deltas = new float[batch_size*activations_size];
//     cudaMalloc(&d_deltas, activations_size*batch_size*sizeof(float));
//     cudaMemcpy(d_activations, activations, activations_size*batch_size*sizeof(float), cudaMemcpyHostToDevice);
//     // float * d_product = transferMatrixToDevice(activations, batch_size, activations_size);
//     // //initialize array for storing predictions of test set on host
//     float* predictions = (float*)malloc(activations_size*batch_size*sizeof(float));
//     float * test_predictions = (float*)malloc(test_size*model->nClasses*sizeof(float));
//     std::shared_ptr<float> d_test_product = transferMatrixToDevice(test_predictions, test_size, model->nClasses);
//     //define metrics
//     int correct = 0;
//     double logLoss = 0.0;
//     float accuracy = 0.0;
//     int totalEpochs = 0;
//     std::cout << "START TRAIN" << std::endl;
//     auto startTrain = std::chrono::system_clock::now();
//     for(int i = 0; i < nEpochs; i++) {
//         correct = 0;
//         logLoss = 0;
//         accuracy = 0.0;
//         for(int j = 0; j < batch_size; j+=batch_size) {
//             //pass inputs through the network
//             // auto startForward = std::chrono::system_clock::now();
//             //let's work on making the dot product function significantly faster
//             /*
//             We can create a dim3 of size (nWorkers, 2, 1) as the number of blocks passed to our dot product function
//             And a dim3 of size(nThreadsPerWorker, 2, 1) as the number of threads per block
//             */
//             // predict<<<nWorkers, nThreadsPerWorker>>>(d_model, d_inputs+(j*model->layer_size[0]), d_activations, d_offsets, batch_size);    
//             cudaDeviceSynchronize();
//             // // auto endForward = std::chrono::system_clock::now();
//             // // std::chrono::duration<double> elapsed_forward = endForward - startForward;
//             // // std::cout << "Finished forward pass in " << elapsed_forward.count() << " seconds" << std::endl;
//             // error = cudaMemcpy(predictions, d_activations, activations_size*batch_size*sizeof(float), cudaMemcpyDeviceToHost);
//             // correct += getAccuracy(predictions+offsets[1], train_labels, batch_size, model->nClasses, j);
//             // logLoss += crossEntropyLoss(predictions+offsets[1], train_labels, batch_size, model->nClasses, j);
//             // // printf("Accuracy: %f%%\n", correct / (float) (batch_size)* 100);
//             // // //compute gradients in forward_pass
//             // // auto startBackward = std::chrono::system_clock::now();
//             // backprop<<<nWorkers, nThreadsPerWorker>>>(d_model, d_inputs+(j*(model->layer_size[0])), d_outputs+(j*(model->nClasses)), d_activations, d_deltas, d_offsets, batch_size, model->nClasses);
//             // cudaDeviceSynchronize();
//             // ringReduce<<<nWorkers, nThreadsPerWorker>>>(d_model, nWorkers*nThreadsPerWorker);
//             // cudaDeviceSynchronize();
//             // // auto endReduce = std::chrono::system_clock::now();
//             // // std::chrono::duration<double> elapsed_reduce = endReduce - startReduce;
//             // // std::cout << "Finished ring reduce in " << elapsed_reduce.count() << " seconds" << std::endl;
//             // // auditDeltas<<<1,1>>>(d_model, d_deltas, d_offsets, nWorkers*nThreadsPerWorker, batch_size);
//             // // cudaDeviceSynchronize();
//             // // auditGradients<<<1,1>>>(d_model);
//             // // cudaDeviceSynchronize();
//             // auto startUpdate = std::chrono::system_clock::now();
//             // backward_pass<<<nWorkers, nThreadsPerWorker>>>(d_model, batch_size, learning_rate);
//             // cudaDeviceSynchronize();
//             // // auditWeights<<<1,1>>>(d_model);
//             // // cudaDeviceSynchronize();
//             // // auto endUpdate = std::chrono::system_clock::now();
//             // // std::chrono::duration<double> elapsed_update = endUpdate - startUpdate;
//             // // std::cout << "Finished weight update in " << elapsed_update.count() << " seconds" << std::endl;
//             // totalEpochs++;
//         }
//         accuracy = correct / (float) total_size;
//         printf("End of epoch %d\n", i+1);
//         printf("Accuracy: %f%%\n", accuracy*100);
//     }
//     auto endTrain = std::chrono::system_clock::now();
//     std::chrono::duration<double> elapsed_forward = endTrain - startTrain;
//     std::cout << "Finished forward pass in " << elapsed_forward.count() << " seconds" << std::endl;
//     for(int i = 1; i < model->nLayers+1; i++) {
//         cudaFree(d_model->weights[i-1]);
//         cudaFree(d_model->biases[i-1]);
//         cudaFree(d_model->gradients[i-1]);
//         // cudaMemcpy(temp_gradients[i-1], model->gradients[i-1], nThreadsPerWorker*nWorkers*model->layer_size[i-1]*model->layer_size[i]*sizeof(float), cudaMemcpyHostToDevice);
//         cudaFree(d_model->grad_biases[i-1]);
//     }
//     cudaFree(d_model->layer_size);
//     cudaFree(d_inputs);
//     cudaFree(d_test_inputs);
//     cudaFree(d_activations);
//     cudaFree(d_deltas);
    
    
// }

