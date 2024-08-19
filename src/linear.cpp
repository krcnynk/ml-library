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

    std::unique_ptr<Tensor> Linear::forward(const Tensor &input,bool transpose)
    {
        Tensor internal_weight_ = weight_;
        if(transpose)
        {
            printVector(internal_weight_.shape());
            internal_weight_ = internal_weight_.transpose();
            printVector(internal_weight_.shape());
        }

        this->input_ = input;
        Tensor output = input_.matmul(internal_weight_); // row vector
        output = output + bias_;

        return std::make_unique<Tensor>(output);
    }

    Tensor Linear::input()
    {
        return input_;
    }

    void Linear::weightUpdate(float learning_rate, Tensor delta)
    {
        // weight_ = weight_ - learning_rate * input_.matmul(delta);
    }
    // std::unique_ptr<Tensor> Linear::backward(const Tensor &input)
    // {
    //     this->input_ = input;
    //     Tensor output = input.matmul(weight_); //row vector
    //     output = output + bias_;

    //     return std::make_unique<Tensor>(output);
    // }

}