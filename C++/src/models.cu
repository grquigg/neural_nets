#include "../include/utils.h"
#include "../include/nn.h"
#include "../include/lin_alg.h"
#include <chrono> 
#include <iostream>
#include <cublas_v2.h>

// LogisticRegression * copyModelToHost(LogisticRegression *model, LogisticRegression *start) {
//     LogisticRegression* host;
//     host->weights = initializeFlatRandomArray(nFeatures, numClasses);
//     host->gradients = (float*)malloc(nFeatures*numClasses*sizeof(float));
// }
