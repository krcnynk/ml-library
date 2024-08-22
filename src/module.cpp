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

    void Module::add_module(const std::string &name, std::shared_ptr<Module> module)
    {
        // assignment operator not made yet for Tensor
        modules_.push_back(std::make_pair(name, module));
    }

    void Module::load_data(const Tensor &input, const Tensor &target)
    {
        input_data_ = input;
        target_ = target;
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

    // float Module::MSE(const Tensor &input, const Tensor &input2)
    // {
    //     // FIXME: Need to add [] operator OR NOT!
    //     float sum = 0;
    //     Tensor intermediary = (input2 - input);
    //     intermediary = intermediary * intermediary; // hadamard
    //     sum = intermediary.sum();

    //     return sum / static_cast<float>(intermediary.size());
    // }

    // std::unique_ptr<Tensor> Module::forward(const Tensor &input, bool tranpose)
    // {
    //     std::unique_ptr<Tensor> output = std::make_unique<Tensor>(input);
    //     for (auto &[name, module] : modules_)
    //     {
    //         output = module->forward(*output); // Call Linear's forward
    //     }
    //     prediction_ = output;
    //     return output;
    // }

    // std::unique_ptr<Tensor> Module::backward(const Tensor &prediction)
    // {

    //     // float learning_rate = 0.001f;
    //     // auto it = modules_.rbegin();
    //     // std::shared_ptr<Module> activation = it->second; // fn-xn
    //     // ++it;
    //     // std::shared_ptr<Module> linear_layer = it->second; // wn-bn
    //     // if (it == modules_.rend())
    //     // {
    //     //     Tensor delta = (prediction - target_) * *activation->backward(linear_layer->forward(input_data_));
    //     // }
    //     // ++it;
    //     // std::shared_ptr<Module> activation_next = it->second; // fn-1-xn-1
    //     // Tensor delta = (prediction - target_) * *activation->backward(linear_layer->forward(activation_next->output()));
    //     // while (it != modules_.rend())
    //     // {

    //     //     ++it;
    //     //     std::shared_ptr<Module> linear_layer_next = it->second; // wn-1-bn-1
    //     //     if (it == modules_.rend())
    //     //     {
    //     //         Tensor delta = linear_layer->forward(delta, true) * *activation_next->backward(linear_layer_next->forward(input_data_));
    //     //         break;
    //     //     }
    //     //     else
    //     //     {
    //     //         ++it;
    //     //         std::shared_ptr<Module> activation_next_next = it->second; // fn-2-xn-2
    //     //         Tensor delta = linear_layer->forward(delta, true) * *activation_next->backward(linear_layer_next->forward(activation_next_next->output()));
    //     //     }
    //     // }
    //     // linear_layer->weightUpdate(learning_rate, delta_i);

    //     return std::make_unique<Tensor>(input_data);
    // }

    std::unique_ptr<Tensor> Module::train(const Tensor &input, const Tensor &target)
    {
        std::unique_ptr<Tensor> x_n;
        // while(true)
        for (int i = 0; i < 50000; ++i)
        {
            x_n = this->forward(input);
            this->backward(*x_n);
            std::cout << this->MSE(*x_n, target) << std::endl;
        }

        x_n = this->forward(input);
        return x_n;
    }

    // Tensor Module::output()
    // {
    //     return prediction_;
    // }

    // void Module::weightUpdate(float learning_rate, Tensor delta)
    // {
    //     // do nothing for Module
    //     return;
    // }

    // void Module::transposeWeights()
    // {
    //     // do nothing for Module
    //     return;
    // }

}