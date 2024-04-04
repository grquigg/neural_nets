# Creating a neural network in (too many) different programming languages #

This is a repository for building a Neural Network in about as many programming languages there are out there. So far, I've only written logistic regression code for Scala, Ruby and C. I'm currently working on implementing a full neural network in C++ and MATLAB, as well as porting over some old Python code from my Machine Learning grad school class.

There are a few overall goals of this project:

1. Showcasing that neural networks CAN be language agnostic,
2. Learning how to program in various different languages that I've never seen before,
3. To examine the various strengths and weaknesses for each language when it comes to a high performance machine learning model that requires intensive amounts of compute and memory,
4. To explore how machine learning frameworks like PyTorch and Tensorflow really work under the hood.  

## Dataset ##

The Dataset I'm primarily using for training all of these neural networks in the MNIST Handwritten Digits dataset. It's a good example that simple neural networks can achieve very good performance on. 

However, another one of the main underlying goals of this project is to ensure that these networks can be reused for other datasets, since I will likely pursue other ML projects in the future using these repositories. 

## C++ ##

There are a few prerequisites to running the C++ code, as the current implementation only works using CUDA currently. Therefore, having a GPU and CUDA installed properly is paramount to this code running. This is  something that will be addressed in later development.

This code also uses Google Tests for test cases. Google Tests requires a minimum of C++ 14. 

Building the code can be done using the following commands:

```
cd C++
mkdir build 
cd build
cmake ..
cmake --build .
```
CMake will default to building the executables in a Debug subdirectory of build. So the path for running executables would be\
```./Debug/main PATH_TO_MNIST_FILES BATCH_SIZE N_WORKERS N_THREADS_PER_WORKER```

```PATH_TO_MNIST_FILES``` indicates the path to where the training and testing files for MNIST are on the local machine\
```BATCH_SIZE``` indicates the batch size you intend to train the network with\
```N_WORKERS``` and ```N_THREADS_PER_WORKER``` are both CUDA specific variables that tell the program how many copies of the model that it should make and run training on a batch on in parallel. If you don't have a super compute intensive GPU then these should be set to relatively small values.

**IMPORTANT**: All three of the numerical values should be a divisor of 60000 (number of training images in the dataset). One of the things on my to do list to make it such that this is more abstracted away from users so that they don't have to worry about it as much. 

This code is very much still a work in progress since I'm making a lot of the code I've written so far less C-like and more "C++-like", taking advantage of a lot of OOP principles and a better understanding of how memory allocation works in CUDA compared to when I first started working on this.

In order to run tests, you can simply run ```ctest``` or any of the individual test suits by calling ```./Debug/[NAME_OF_TEST_SUITE]```
## Scala ##

The command to compile and run the Scala code is 
```
scalac main.scala
scala Network ../mnist/train-images.idx3-ubyte ../mnist/train-labels.idx1-ubyte
```

## MATLAB ##

Only the test portion of the MATLAB code is actually fully functional at this stage in development. The test cases in ```testing.m``` should all pass. 