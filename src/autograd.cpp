#include "autograd.h"
#include <stdexcept>

namespace ml_framework
{
    std::shared_ptr<Tensor> AddOperation::forward(const std::vector<std::shared_ptr<Tensor>> &inputs)
    {
        if (inputs.size() != 2)
            throw std::runtime_error("AddOperation expects exactly two inputs.");

        const auto &a = inputs[0];
        const auto &b = inputs[1];

        if (a->shape() != b->shape())
            throw std::runtime_error("Tensors must have the same shape for addition.");

        std::shared_ptr<Tensor> result_tensor = std::make_shared<Tensor>(a->shape());
        const float alpha = 1.0f;
        cublasStatus_t status = cublasSaxpy(a->cublas_handle, static_cast<int>(a->size()), &alpha, b->device_data(), 1, result_tensor->device_data(), 1);
        CHECK_CUBLAS_STATUS(status);
        result_tensor->transferDataToHost();
        return result_tensor;
    }

    std::vector<std::shared_ptr<Tensor>> AddOperation::backward(const std::shared_ptr<Tensor> &grad_output)
    {
        // Compute gradients for addition
        std::shared_ptr<Tensor> grad_a = std::make_shared<Tensor>(grad_output->shape());
        std::shared_ptr<Tensor> grad_b = std::make_shared<Tensor>(grad_output->shape());

        // Gradients for addition
        // grad_a->data = grad_output->data;
        // grad_b->data = grad_output->data;

        return std::vector<std::shared_ptr<Tensor>>{grad_a, grad_b};
    }
}
