#ifndef UTILS_H
#define UTILS_H
#include <vector>
#include <fstream>

int convertCharsToInt(std::vector<char> chars);

std::vector<std::vector<int>> readDataFromUByteFile(std::string filePath);

std::vector<float> initializeRandomArray(int mat_height, int mat_width);

void printMatrix(float * arr, int height, int width);

void printMatrix(std::vector<float> arr, int height, int width);

#endif