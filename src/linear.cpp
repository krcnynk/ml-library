#include "linear.h"

namespace ml_framework
{
    Linear::Linear(int input_features, int output_features)
    {
        const std::vector<int> weight_shape{input_features, output_features};
        weight_ = Tensor(weight_shape);
        std::vector<int> bias_shape{1, output_features};
        // Initialize bias tensor with shape (out_features)
        bias_ = Tensor(bias_shape);
    }

    std::unique_ptr<Tensor> Linear::forward(const Tensor &input)
    {
        Tensor output = input.matmul(weight_);
        output = output + bias_;

        return std::make_unique<Tensor>(output);
    }

}