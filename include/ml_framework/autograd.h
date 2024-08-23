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
        virtual Tensor forward(const Tensor &a, const Tensor &b) = 0;
        // Perform the backward pass, returning the gradients w.r.t inputs
        virtual std::vector<Tensor> backward(const std::shared_ptr<Tensor> &grad_output) = 0;
    };

    class AddOperation : public Operation
    {
    public:
        // Override the forward pass for addition
        Tensor forward(const Tensor &a, const Tensor &b) override;

        // Override the backward pass for addition
        std::vector<Tensor> backward(const std::shared_ptr<Tensor> &grad_output) override;
    };
}

#endif // AUTOGRAD_H
