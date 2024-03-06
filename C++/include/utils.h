#ifndef UTILS_H
#define UTILS_H
#include <vector>
#include <fstream>

int convertCharsToInt(std::vector<char> chars);

std::vector<std::vector<int>> readDataFromUByteFile(std::string filePath);

std::vector<float> initializeRandomArray(int mat_height, int mat_width);

float * initializeFlatRandomArray(int mat_height, int mat_width);

void printMatrix(float * arr, int height, int width);

void printMatrix(std::vector<float> arr, int height, int width);

int getAccuracy(float* predicted, std::vector<std::vector<int>> actual, int height, int width, int index);

double crossEntropyLoss(float* predicted, std::vector<std::vector<int>>& actual, int height, int width, int index);
#endif