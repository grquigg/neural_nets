#include "../include/utils.h"
#include <vector>
#include <iostream>
#include <cassert>
#include <stdlib.h>
#include <cmath>

// int convertCharsToInt(std::vector<char>& chars, int begin, int end) {
//     int d = 0;
//     int n = end - begin;
//     int power = 0;
//     for(int i = begin; i < end; i++) {
//         d |= chars[end-(i-begin)-1] << 2**0;
//         power += 2;
//     }
//     return d;
// }

std::vector<float> initializeRandomArray(int mat_height, int mat_width) {
    std::vector<float> weights(mat_height*mat_width, 0.0);
    int a = 1;
    for (int i = 0; i < mat_height; i++) {
        for(int j = 0; j < mat_width; j++) {
            // weights[i*mat_width+j] = (float)rand()/(float)(RAND_MAX/a);
            weights[i*mat_width+j] = 0.25;
            //the most important line in the entire program
        }
    }
    return weights;
}

float* initializeFlatRandomArray(int mat_height, int mat_width) {
    float * arr = (float *)malloc(mat_height * mat_width * sizeof(float));
    for(int i = 0; i < mat_height; i++) {
        for(int j = 0; j < mat_width; j++) {
            arr[(i*mat_width)+j] = 1/(float) ((i*mat_width)+j+1);
        }
    }
    return arr;
}

std::vector<std::vector<int>> readDataFromUByteFile(std::string filePath) {
    std::ifstream input(filePath, std::ios::binary);
    //load all bytes into a buffer
    std::vector<unsigned char> buffer(std::istreambuf_iterator<char>(input), {});
    //first two bytes are zero, so we take the third byte and assert that it's eight
    int buffer_ptr = 0;
    int dims = buffer[3];
    std::vector<int> dimensions;
    for(int i = 0; i < dims; i++) {
        int d = buffer[4+(i*4)] << 24 | buffer[4+(i*4)+1] << 16 | buffer[4+(i*4)+2] << 8 | buffer[7+(i*4)];
        dimensions.push_back(d);
    }
    buffer_ptr = 4 + (dims*4);
    // assert(dimensions[0] == 60000);
    // assert(dimensions[1] == 28);
    // assert(dimensions[2] == 28);
    // assert(buffer_ptr == 16);
    //we flatten down the last n-1 dimensions down into a single one
    int second = 1;
    for(int i = 1; i < dimensions.size(); i++) {
        second*= dimensions[i];
    }
    //this will never be resized, thankfully
    //arr is the variable we return once we initialize it with the values from our file
    //iniaitlize values
    std::vector<std::vector<int>> arr(60000, std::vector<int>(second, 0));
    for(int i = 0; i < dimensions[0]; i++) {
        for(int j = 0; j < second; j++) {
            arr[i][j] = buffer[buffer_ptr];
            buffer_ptr++;
        }
    }
    return arr;
}

void printMatrix(float * arr, int height, int width) {
    for(int i = 0; i < height; i++) {
        for(int j = 0; j < width; j++) {
            std::cout << arr[i*width+j] << "\t";
        }
        std::cout << std::endl;
    }
}

int getAccuracy(float* predicted, std::vector<std::vector<int>> actual, int height, int width, int index) {
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
        }
        if ((int) actual[i+index][0] == max) {
            correct++;
        }
    }
    return correct;
}

void printMatrix(std::vector<float> arr, int height, int width) {
    for(int i = 0; i < height; i++) {
        for(int j = 0; j < width; j++) {
            std::cout << arr[i*width+j] << "\t";
        }
        std::cout << std::endl;
    }
}

float crossEntropyLoss(float* predicted, std::vector<std::vector<int>> actual, int height, int width, int index) {
    float log_sum = 0;
    for (int i = 0; i < height; i++) {
        int prediction = actual[index+i][0];
        if(predicted[i*width+prediction] > 1.0e-33) {
            log_sum -= (log(predicted[i*width+prediction]));
        }
    }
    return log_sum;
}