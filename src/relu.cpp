#include "relu.h"
#include "config.h"
#include "tensor.h"

namespace ml_framework
{

    std::unique_ptr<Tensor> ReLU::forward(const Tensor &input)
    {
        constexpr float gradient = -0.5;
        std::cout << "forwarding relu" << std::endl;
        Tensor output = Tensor(input);
        cudaError_t error = leakyReluKernelWrapper(input.device_data(), output.device_data(), gradient, static_cast<int>(input.size()));
        CHECK_CUDA_ERROR(error);
        output.transferDataToHost();
        return std::make_unique<Tensor>(output);
    }

}