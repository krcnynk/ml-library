// For me The linear component typically represents a linear (or fully connected) layer in a neural network. Key features include:
#ifndef ML_FRAMEWORK_LINEAR_H
#define ML_FRAMEWORK_LINEAR_H

#include <iostream>
#include <memory>
#include <vector>
#include "tensor.h"
#include "module.h"

namespace ml_framework
{
  class Linear : public Module
  {
  public:
    Linear() = default;
    Linear(int in_features, int out_features);
    std::unique_ptr<Tensor> forward(const Tensor &input) override;
    ~Linear() override = default;
  private:
    Tensor weight_; // Weight tensor
    Tensor bias_;   // Bias tensor
  };

}
#endif