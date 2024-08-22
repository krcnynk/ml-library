#ifndef MODULE_H
#define MODULE_H

#include <vector>
#include <string>
#include <memory>
#include <stdexcept>
#include <list>
#include "tensor.h"

namespace ml_framework
{
    class Module
    {
    public:
        Module() = default; // calls list()
        virtual ~Module() = default;
        void add_module(const std::string &name, std::shared_ptr<Module> module);
        std::shared_ptr<Module> get_module(const std::string &name) const;
        void load_data(const Tensor &input, const Tensor &target);
        std::unique_ptr<Tensor> train(const Tensor &input, const Tensor &target);
        // virtual Tensor forward(const Tensor &input) = 0;
        // void register_parameter(const std::string &name, const Tensor &tensor);

        // float MSE(const Tensor &input, const Tensor &input2);
        // virtual std::unique_ptr<Tensor> forward(const Tensor &input, bool transpose = false);
        // virtual std::unique_ptr<Tensor> backward(const Tensor &prediction);
        // virtual Tensor output();
        // virtual void weightUpdate(float learning_rate, Tensor delta);

    private:
        Tensor input_data_;
        Tensor target_;
        Tensor prediction_;
        std::list<std::pair<std::string, std::shared_ptr<Layer>>> layers_; // Submodules
    };
}

#endif