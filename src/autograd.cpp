#include "autograd.h"
#include <stdexcept>
#include "config.h"
namespace ml_framework
{
    std::unique_ptr<Tensor> AddOperation::forward(const Tensor &a, const Tensor &b)
    {
        if (a.shape() != b.shape())
        {
            throw std::runtime_error("Tensor m_shapes must match for addition.");
        }

        // Tensor result{a};
        auto result = std::make_unique<Tensor>(a);
        const float alpha = 1.0f;

        cublasStatus_t status = cublasSaxpy(Tensor::cublas_handle, static_cast<int>(a.size()),
                                            &alpha, b.device_data(), 1, result->device_data(), 1);
        CHECK_CUBLAS_STATUS(status);
        result->transferDataToHost();
        return result; // has move constructor
    }

    std::vector<std::unique_ptr<Tensor>> AddOperation::backward(const std::unique_ptr<Tensor> &grad_output,
                                                                const Tensor &a, const Tensor &b)
    {
        // Accumulate gradients
        if (a.grad == nullptr || b.grad == nullptr)
        {
            throw std::runtime_error("Gradient not initialized before accumulation.");
        }

        Tensor grad_a_pass{*grad_output}; // making new Tensors
        Tensor grad_b_pass{*grad_output};
        std::unique_ptr<Tensor> up_grad_a = forward(*a.grad, *grad_output);
        std::unique_ptr<Tensor> up_grad_b = forward(*b.grad, *grad_output);

        std::vector<std::unique_ptr<Tensor>> result;
        result.push_back(std::move(up_grad_a));
        result.push_back(std::move(up_grad_b));
        return result;
    }

    std::unique_ptr<Tensor> MultiplyOperation::forward(const Tensor &a, const Tensor &b)
    {
        if (a.shape() != b.shape())
        {
            throw std::runtime_error("Tensor shapes must match for multiplication.");
        }

        auto result = std::make_unique<Tensor>(a);
        cudaError_t err = elementWiseMultiplyKernelWrapper(a.device_data(), b.device_data(), result->device_data(), static_cast<int>(a.size()));
        CHECK_CUDA_ERROR(err);
        result->transferDataToHost();
        return result;
    }

    std::vector<std::unique_ptr<Tensor>> MultiplyOperation::backward(const std::unique_ptr<Tensor> &grad_output,
                                                    const Tensor &a, const Tensor &b)
    {
        // Accumulate gradients
        if (a.grad == nullptr || b.grad == nullptr)
        {
            throw std::runtime_error("Gradient not initialized before accumulation.");
        }

        auto add_operation = AddOperation();

        auto grad_a = forward(*grad_output,b);
        auto grad_b = forward(*grad_output,a);
        std::unique_ptr<Tensor> up_grad_a = add_operation.forward(*a.grad, *grad_a);
        std::unique_ptr<Tensor> up_grad_b = add_operation.forward(*b.grad, *grad_b);

        std::vector<std::unique_ptr<Tensor>> result;
        result.push_back(std::move(up_grad_a));
        result.push_back(std::move(up_grad_b));
        return result;
    }

     std::unique_ptr<Tensor> SubOperation::forward(const Tensor &a, const Tensor &b)
    {
        if (a.shape() != b.shape())
        {
            throw std::runtime_error("Tensor m_shapes must match for addition.");
        }

        // Tensor result{a};
        auto result = std::make_unique<Tensor>(a);
        const float alpha = -1.0f;

        cublasStatus_t status = cublasSaxpy(Tensor::cublas_handle, static_cast<int>(a.size()),
                                            &alpha, b.device_data(), 1, result->device_data(), 1);
        CHECK_CUBLAS_STATUS(status);
        result->transferDataToHost();
        return result; // has move constructor
    }

    std::vector<std::unique_ptr<Tensor>> SubOperation::backward(const std::unique_ptr<Tensor> &grad_output,
                                                                const Tensor &a, const Tensor &b)
    {
        // Accumulate gradients
        if (a.grad == nullptr || b.grad == nullptr)
        {
            throw std::runtime_error("Gradient not initialized before accumulation.");
        }

        auto sub_operation = SubOperation();

        Tensor grad_a_pass{*grad_output}; // making new Tensors
        Tensor grad_b_pass{*grad_output};
        std::unique_ptr<Tensor> up_grad_a = sub_operation.forward(*a.grad, *grad_output);
        std::unique_ptr<Tensor> up_grad_b = sub_operation.forward(*b.grad, *grad_output);

        std::vector<std::unique_ptr<Tensor>> result;
        result.push_back(std::move(up_grad_a));
        result.push_back(std::move(up_grad_b));
        return result;
    }

}
