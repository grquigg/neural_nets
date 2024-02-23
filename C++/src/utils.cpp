#include "utils.h"
#include <vector>
#include <iostream>
#include <cassert>
#include <stdlib.h>

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

std::vector<std::vector<float>> initializeRandomArray(int mat_height, int mat_width) {
    std::vector<std::vector<float>> weights(mat_height, std::vector<float>(mat_width, 0.0));
    int a = 1;
    for (int i = 0; i < mat_height; i++) {
        for(int j = 0; j < mat_width; j++) {
            weights[i][j] = (float)rand()/(float)(RAND_MAX/a);
            // weights[i][j] = 0.5;
            //the most important line in the entire program
        }
    }
    return weights;
}

std::vector<std::vector<int>> readDataFromUByteFile(std::string filePath) {
    std::ifstream input(filePath, std::ios::binary);
    //load all bytes into a buffer
    std::vector<unsigned char> buffer(std::istreambuf_iterator<char>(input), {});
    //first two bytes are zero, so we take the third byte and assert that it's eight
    int buffer_ptr = 0;
    int dims = buffer[3];
    std::cout << dims << std::endl;
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