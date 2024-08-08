// For me The linear component typically represents a linear (or fully connected) layer in a neural network. Key features include:
#ifndef ML_FRAMEWORK_LINEAR_H
#define ML_FRAMEWORK_LINEAR_H

#include <vector>
#include <iostream>
#include <random>
#include "ml_framework/tensor.h"

namespace nn
{
  class Linear
  {
    // Implementation of a linear (fully connected) layer
    Linear(size_t input_features, size_t output_features);

    ml_framework::Tensor forward(const ml_framework::Tensor &input);
    void backward(const ml_framework::Tensor &grad_output);

    ml_framework::Tensor getWeights() const;
    void setWeights(const ml_framework::Tensor &weights);
    ml_framework::Tensor getBiases() const;
    void setBiases(const ml_framework::Tensor &biases);
    void initializeWeights();
  };
}

#endif