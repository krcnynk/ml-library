#include "module.h"
#include "linear.h"
#include "relu.h"
#include "tensor.h"
#include <typeinfo>

namespace ml_framework
{
    // void Module::register_parameter(const std::string &name,const Tensor &tensor)
    // {
    //     parameters_[name] = &tensor;
    // }

    // Tensor &Module::get_parameter(const std::string &name) const
    // {
    //     return *parameters_.at(name);
    // }

    void Module::add_module(const std::string &name, std::shared_ptr<Module> module)
    {
        // assignment operator not made yet for Tensor
        modules_.push_back(std::make_pair(name, module));
    }

    std::shared_ptr<Module> Module::get_module(const std::string &name) const
    {
        for (const auto &pair : modules_)
        {
            if (pair.first == name)
            {
                return pair.second;
            }
        }
        throw std::invalid_argument("Module with name " + name + " not found.");
    }

    float Module::MSE(const Tensor &input, const Tensor &input2)
    {
        // FIXME: Need to add [] operator OR NOT!
        float sum = 0;
        Tensor intermediary = (input2 - input);
        intermediary = intermediary * intermediary; // hadamard
        sum = intermediary.sum();

        return sum / static_cast<float>(intermediary.size());
    }
    // std::vector<Tensor *> Module::parameters()
    // {
    //     std::vector<Tensor *> params;

    //     for (auto &[name, param] : parameters_)
    //     {
    //         params.push_back(param);
    //     }
    //     for (auto &[name, module] : modules_)
    //     {
    //         auto sub_params = module->parameters();
    //         params.insert(params.end(), sub_params.begin(), sub_params.end());
    //     }
    //     return params;
    // }

    std::unique_ptr<Tensor> Module::forward(const Tensor &input, bool tranpose)
    {
        input_ = input;
        std::unique_ptr<Tensor> output = std::make_unique<Tensor>(input);
        for (auto &[name, module] : modules_)
        {
            output = module->forward(*output); // Call Linear's forward
        }
        return output;
    }

    std::unique_ptr<Tensor> Module::backward(const Tensor &prediction)
    {

        // Tensor x_n = prediction;       // after activation layer, x3, row vector
        // Tensor delta_i = x_n - target; // Initial delta_i for the output layer
        // Tensor gradient;

        // // Iterate in reverse through the modules list
        // auto it = modules_.rbegin(); // Reverse iterator starting from the last module

        // while (it != modules_.rend())
        // {
        //     std::shared_ptr<Module> linear_layer = it->second; // Linear layer
        //     gradient = linear_layer->backward(delta_i);        // Backward pass through the linear layer
        //     ++it;

        //     if (it != modules_.rend())
        //     {
        //         std::shared_ptr<Module> activation = it->second; // Activation function layer
        //         delta_i = *activation->backward(gradient);       // Backward pass through the activation function
        //         ++it;
        //     }
        // }

        auto it = modules_.rbegin();
        std::shared_ptr<Module> activation = it->second;
        ++it;
        std::shared_ptr<Module> linear_layer = it->second;
        Tensor delta_i = (prediction - input_) * *(activation->backward(*linear_layer->forward(linear_layer->input())));
        // linear_layer->weightUpdate(0.001,delta_i);
        while (it != modules_.rend())
        {
            ++it;
            activation = it->second;
            if (it != modules_.rend())
            {
                ++it;
                linear_layer = it->second;
                std::cout << "printing delta" << std::endl;
                printVector(delta_i.shape());
                printVector((activation->backward(*linear_layer->forward(linear_layer->input())))->shape());
                //FIXME: NEED TRANSPOSING LINEAR LAYER WEIGHTS AND DELETE THE INPUT()
                delta_i = *linear_layer->forward(delta_i,true);

                if (it != modules_.rend())
                {
                    ++it;
                    delta_i * *(activation->backward(*linear_layer->forward(linear_layer->input())))
                }
                
                std::cout << " endl" << std::endl;
                // linear_layer->weightUpdate(0.001,delta_i);
            }
        }
        return std::make_unique<Tensor>(input_);
    }

    std::unique_ptr<Tensor> Module::train(const Tensor &input, const Tensor &target)
    {
        std::unique_ptr<Tensor> x_n;
        // while(true)
        for (int i = 0; i < 1000000; ++i)
        {
            x_n = this->forward(input);
            this->backward(*x_n);
        }

        x_n = this->forward(input);
        std::cout << this->MSE(*x_n, target) << std::endl;
        return x_n;
    }

    Tensor Module::input()
    {
        return input_;
    }

    void Module::weightUpdate(float learning_rate, Tensor delta)
    {
        // do nothing for Module
        return;
    }

}