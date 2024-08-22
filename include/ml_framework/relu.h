#ifndef RELU_H
#define RELU_H

#include <memory>
#include "module.h" // Include the base Module class
#include "tensor.h"
#include "layer.h"

namespace ml_framework
{

    class ReLU : public Layer
    {
    public:
        ReLU() = default;
        ~ReLU() override = default;
        std::unique_ptr<Tensor> forward(const Tensor &input, bool transpose = false) override;
        std::unique_ptr<Tensor> backward(const Tensor &input) override;
        Tensor output() override;
    };

}

#endif
