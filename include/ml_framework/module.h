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

        // virtual Tensor forward(const Tensor &input) = 0;
        // void register_parameter(const std::string &name, const Tensor &tensor);
        void add_module(const std::string &name, std::shared_ptr<Module> module);
        // Tensor &get_parameter(const std::string &name) const;
        std::shared_ptr<Module> get_module(const std::string &name) const;
        // FIXME: Double check this!
        // std::vector<Tensor *> parameters();
        virtual ~Module() = default;
        virtual std::unique_ptr<Tensor> forward(const Tensor &input);

    protected:
        // std::unordered_map<std::string, Tensor *> parameters_;
        std::list<std::pair<std::string, std::shared_ptr<Module>>> modules_; // Submodules
    };
}
#endif