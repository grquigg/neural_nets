#include <iostream>
#include <cassert>
#include <string>
#include <vector>
#include "utils.h"
#include "lin_alg.h"

using namespace std;

int main(int argc, char** argv) {
    int numClasses = 10;
    std::cout << "Hello World!" << std::endl;
    std::cout << "Train data path: " << argv[1] << std::endl;
    if(argc != 3) {
        std::cout << "Need to specify paths for loading in the training data and the training labels" << std::endl;
        return 0;
    }
    string train_data_path = argv[1];
    string train_label_path = argv[2];
    std::vector<std::vector<int>> inputs = readDataFromUByteFile(train_data_path);
    std::vector<std::vector<float>> input(60000, std::vector<float>(784, 0.0f));
    for(int i = 0; i < input.size(); i++) {
        for(int j = 0; j < input[0].size(); j++) {
            input[i][j] = (float) inputs[i][j] / 255;
        }
    }
    std::vector<std::vector<int>> outputs = readDataFromUByteFile(train_label_path);
    std::vector<std::vector<float>> weights = initializeRandomArray((int) inputs[0].size(), numClasses);
    std::vector<std::vector<float>> product(784, std::vector<float>(10, 0.0));
    float learning_rate = 0.005;
    int BATCH_SIZE = 1000;
    dotProduct(input, weights, &product);
    return 0;
}