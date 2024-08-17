#ifndef ML_FRAMEWORK_MODULE_H
#define ML_FRAMEWORK_MODULE_H

#include <vector>
#include <unordered_map>
#include <string>
#include <memory>
#include <stdexcept>
#include "tensor.h"

namespace ml_framework
{
    class Module
    {
    public:
        virtual Module() = default;
        virtual ~Module() = default;
        virtual Tensor forward(const Tensor &input) = 0;

        void register_parameter(const std::string &name, Tensor &tensor)
        {
            parameters_[name] = &tensor;
        }

         Tensor get_parameter(const std::string &name) const {
        return *parameters_.at(name);
    }

        // Method to add a submodule to this module
        void add_module(const std::string &name, std::shared_ptr<Module> module)
        {
            modules_[name] = std::move(module);
        }

        // Method to retrieve a submodule by name
        std::shared_ptr<Module> get_module(const std::string &name) const
        {
            auto it = modules_.find(name);
            if (it != modules_.end())
            {
                return it->second;
            }
            throw std::invalid_argument("Module with name " + name + " not found.");
        }

        // Method to collect all parameters from this module and its submodules
        std::vector<Tensor *> parameters()
        {
            std::vector<Tensor *> params;
            for (auto &param : parameters_)
            {
                params.push_back(param);
            }
            for (auto &[name, module] : modules_)
            {
                auto sub_params = module->parameters();
                params.insert(params.end(), sub_params.begin(), sub_params.end());
            }
            return params;
        }

    protected:
        std::unordered_map<std::string, Tensor *> parameters_;
        std::unordered_map<std::string, std::shared_ptr<Module>> modules_; // Submodules
        bool is_training_ = true;

        // Method to register a parameter (e.g., weights, biases)
        // void register_parameter(const std::string &name, Tensor &tensor)
        // {
        //     parameters_.push_back(&tensor);
        // }

    // private:
    //     std::unordered_map<std::string, Tensor *> parameters_;
    //     bool is_training_ = true;

    //     std::vector<Tensor *> parameters_;                                 // Parameters in this module
    //     std::unordered_map<std::string, std::shared_ptr<Module>> modules_; // Submodules
    // };

}

#endif