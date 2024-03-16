#include <torch/torch.h>
#include <iostream>
#include <c10d/ProcessGroupMPI.hpp>

struct Network : torch::nn::Module {
  torch::nn::Linear lin1;
  torch::nn::Linear lin2;
}

int main() {
  torch::DeviceType device_type = torch::kCUDA;

  if(torch::cuda::is_available()) {
    std::cout << "CUDA is available" << std::endl;
  }
}