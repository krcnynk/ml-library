#ifndef LAYER_H
#define LAYER_H

#include <vector>
#include <string>
#include <memory>
#include <stdexcept>
#include <list>
#include "tensor.h"

namespace ml_framework
{
    class Layer
    {
    public:
        Layer() = default; // calls list()
        virtual ~Layer() = default;
        virtual std::unique_ptr<Tensor> forward(const Tensor &input, bool transpose = false) = 0;
        virtual std::unique_ptr<Tensor> backward(const Tensor &prediction) = 0;
    protected:
        // std::unordered_map<std::string, Tensor *> parameters_;
        Tensor input_layer;
        Tensor output_layer;
    };
}

#endif