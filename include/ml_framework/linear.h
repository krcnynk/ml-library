// For me The linear component typically represents a linear (or fully connected) layer in a neural network. Key features include:
#ifndef LINEAR_H
#define LINEAR_H

#include <iostream>
#include <memory>
#include <vector>
#include "tensor.h"
#include "module.h"
#include "layer.h"

namespace ml_framework
{
    class Linear : public Layer
    {
    public:
        Linear() = default;
        ~Linear() override = default;
        Linear(int in_features, int out_features);
        std::unique_ptr<Tensor> forward(const Tensor &input, bool tranpose = false) override;
        // Tensor output();
        // void weightUpdate(float learning_rate, Tensor delta) override;
        // void transposeWeights();
        // std::unique_ptr<Tensor> backward(const Tensor &input) override;

    private:
        Tensor weight_; // Weight tensor
        Tensor bias_;   // Bias tensor
    };
}

#endif