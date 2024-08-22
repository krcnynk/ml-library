#ifndef AUTOGRAD_H
#define AUTOGRAD_H

#include <memory>
#include <vector>
#include "tensor.h"

namespace ml_framework
{
    class Operation
    {
    public:
        virtual ~Operation() = default;

        // Perform the forward pass for this operation
        virtual std::shared_ptr<Tensor> forward(const std::vector<std::shared_ptr<Tensor>> &inputs) = 0;
        // Perform the backward pass, returning the gradients w.r.t inputs
        virtual std::vector<std::shared_ptr<Tensor>> backward(const std::shared_ptr<Tensor> &grad_output) = 0;
    };

    class AddOperation : public Operation
    {
    public:
        // Override the forward pass for addition
        std::shared_ptr<Tensor> forward(const std::vector<std::shared_ptr<Tensor>> &inputs) override;

        // Override the backward pass for addition
        std::vector<std::shared_ptr<Tensor>> backward(const std::shared_ptr<Tensor> &grad_output) override;
    };
}

#endif // AUTOGRAD_H
