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
        virtual std::unique_ptr<Tensor> forward(const Tensor &a, const Tensor &b) = 0;
        // Perform the backward pass, returning the gradients w.r.t inputs
        virtual std::vector<std::unique_ptr<Tensor>> backward(const std::unique_ptr<Tensor> &grad_output, const Tensor &a, const Tensor &b) = 0;
    };

    class AddOperation : public Operation
    {
    public:
        // unique_ptr the forward pass for addition
        std::unique_ptr<Tensor> forward(const Tensor &a, const Tensor &b) override;

        // Override the backward pass for addition
        std::vector<std::unique_ptr<Tensor>> backward(const std::unique_ptr<Tensor> &grad_output, const Tensor &a, const Tensor &b) override;
    };

    class MultiplyOperation : public Operation
    {
    public:
        // Unique pointer for the forward pass of multiplication
        std::unique_ptr<Tensor> forward(const Tensor &a, const Tensor &b) override;

        // Override the backward pass for multiplication
        std::vector<std::unique_ptr<Tensor>> backward(const std::unique_ptr<Tensor> &grad_output, const Tensor &a, const Tensor &b) override;
    };

    class SubOperation : public Operation
    {
    public:
        // unique_ptr the forward pass for addition
        std::unique_ptr<Tensor> forward(const Tensor &a, const Tensor &b) override;

        // Override the backward pass for addition
        std::vector<std::unique_ptr<Tensor>> backward(const std::unique_ptr<Tensor> &grad_output, const Tensor &a, const Tensor &b) override;
    };

}

#endif // AUTOGRAD_H
