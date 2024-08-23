#include "autograd.h"
#include <stdexcept>
#include "config.h"
namespace ml_framework
{
    Tensor AddOperation::forward(const Tensor &a, const Tensor &b)
    {
        if (a.shape() != b.shape())
        {
            throw std::runtime_error("Tensor m_shapes must match for addition.");
        }

        Tensor result_tensor(a.m_shape, a.h_data.get());
        const float alpha = 1.0f;
        cublasStatus_t status = cublasSaxpy(result_tensor.cublas_handle, static_cast<int>(a.size()), &alpha, b.device_data(), 1, result_tensor.device_data(), 1);
        CHECK_CUBLAS_STATUS(status);
        result_tensor.transferDataToHost();
        return result_tensor;
    }

    std::vector<Tensor> AddOperation::backward(const std::shared_ptr<Tensor> &grad_output)
    {
        // // addition pass gradients as is
        // return std::vector<Tensor>{
        //     Tensor(grad_output->m_shape, grad_output->h_data.get()),
        //     Tensor(grad_output->m_shape, grad_output->h_data.get())};
    }
}
