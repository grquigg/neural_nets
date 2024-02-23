#ifndef UTILS_H
#define UTILS_H
#include <string>
#include <vector>
#include <fstream>

int convertCharsToInt(std::vector<char> chars);

std::vector<std::vector<int>> readDataFromUByteFile(std::string filePath);

std::vector<std::vector<float>> initializeRandomArray(int mat_height, int mat_width);
#endif