#include "module.h"
#include "linear.h"
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

    std::unique_ptr<Tensor> Module::forward(const Tensor &input)
    {
        std::unique_ptr<Tensor> output = std::make_unique<Tensor>(input);
        for (auto &pair : modules_)
        {
            if (auto linearPtr = std::dynamic_pointer_cast<Linear>(pair.second))
            {
                std::cout << "aaa free" << std::endl;
                output = linearPtr->forward(*output); // Call Linear's forward
            }
            else
            {
                // Do nothing for the time being
            }
        }
        return output;
    }

}