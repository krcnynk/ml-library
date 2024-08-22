#include "relu.h"
#include "config.h"
#include "tensor.h"

namespace ml_framework
{

    std::unique_ptr<Tensor> ReLU::forward(const Tensor &input, bool transpose)
    {
        constexpr float gradient = -0.5;
        Tensor output = input;
        cudaError_t error = leakyReluKernelWrapper(input.device_data(), output.device_data(), gradient, static_cast<int>(input.size()));
        CHECK_CUDA_ERROR(error);
        output.transferDataToHost();
        return std::make_unique<Tensor>(output);
    }

    std::unique_ptr<Tensor> ReLU::backward(const Tensor &input)
    {
        constexpr float gradient = -0.5;
        Tensor output = Tensor(input);
        cudaError_t error = d_leakyReluKernelWrapper(input.device_data(), output.device_data(), gradient, static_cast<int>(input.size()));
        CHECK_CUDA_ERROR(error);
        output.transferDataToHost();
        return std::make_unique<Tensor>(output);
    }

    Tensor ReLU::output()
    {
        return output_;
    }


}