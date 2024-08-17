#include "linear.h"

namespace ml_framework
{
    Linear::Linear(int input_features, int output_features)
    {
        std::vector<int> weight_shape{input_features, output_features};
        // Initialize weight tensor with shape (out_features, in_features)
        weight_ = Tensor(weight_shape);

        std::vector<int> bias_shape{1, input_features};
        // Initialize bias tensor with shape (out_features)
        bias_ = Tensor(bias_shape);

        // Register weight and bias as parameters
        register_parameter("weight", weight_);
        register_parameter("bias", bias_);
    }

    Tensor Linear::forward(const Tensor &input)
    {

        Tensor output = input.matmul(weight_);
        // Add the bias to each row of the output
        output = output + bias_;

        return output;
    }
}