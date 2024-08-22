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

    std::unique_ptr<Tensor> Linear::forward(const Tensor &input, bool transpose)
    {
        Tensor dummy_weight = weight_;
        if (transpose)
        {
            dummy_weight = dummy_weight.transpose();
        }
        Tensor output = input.matmul(dummy_weight) + bias_; // row vector
        return std::make_unique<Tensor>(output + bias_);
    }

    // void Linear::weightUpdate(float learning_rate, Tensor delta)
    // {
    //     std::cout << weight_ << std::endl;
    //     Tensor tranposed_input = input_.transpose();
    //     weight_ = weight_ - learning_rate * tranposed_input.matmul(delta);
    //     std::cout << weight_ << std::endl;
    // }

    // void Linear::transposeWeights()
    // {
    //     weight_ = weight_.transpose();
    // }

    // std::unique_ptr<Tensor> Linear::backward(const Tensor &input)
    // {
    //     this->input_ = input;
    //     Tensor output = input.matmul(weight_); //row vector
    //     output = output + bias_;

    //     return std::make_unique<Tensor>(output);
    // }

}