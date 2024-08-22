#ifndef TENSOR_H
#define TENSOR_H

#include "suppress_warnings.h"
SUPPRESS_SIGN_CONVERSION_WARNINGS
RESTORE_WARNINGS

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <vector>
#include <stdexcept>
#include <iostream>
#include <iomanip>
#include <memory>
#include <algorithm>
#include <random>
#include "config.h"
#include <functional>
// #include "autograd.h"

namespace ml_framework
{
    class Tensor : public std::enable_shared_from_this<Tensor>
    {
    public:
        Tensor();
        Tensor(const std::vector<int> &shape, const float *data);
        Tensor(const Tensor &tensor);
        Tensor(const std::vector<int> &shape);
        Tensor(const std::vector<int> &shape, float init);
        ~Tensor();

        const std::vector<int> &shape() const;
        size_t size() const;
        // int size() const;
        float *host_data() const;
        float *host_data();
        float *device_data() const;
        float *device_data();

        Tensor operator+(const Tensor &other) const;
        Tensor operator-(const Tensor &other) const;
        Tensor operator*(const Tensor &other) const;
        Tensor transpose() const;
        Tensor matmul(const Tensor &other) const;
        Tensor &operator=(const Tensor &other);
        friend Tensor operator*(float scalar, const Tensor &tensor);
        friend std::ostream &operator<<(std::ostream &os, const Tensor &point);

        float sum() const;
        bool operator==(const Tensor &other) const;
        void transferDataToDevice() const;
        void transferDataToHost() const;

        static void initializeCuBLAS();
        static void cleanupCuBLAS();

    private:
        std::vector<int> m_shape;
        std::shared_ptr<float[]> h_data;
        // CUDA resources
        mutable float *d_data = nullptr;     // Device pointer for data
        static cublasHandle_t cublas_handle; // cuBLAS handle
        // Helper functions
        void allocateDeviceMemory() const;
        void freeDeviceMemory() const;
        // KernelCalls
        friend class Operation;
        friend class AddOperation;
        friend class MultiplyOperation;

    public:
        void backward();
        mutable std::shared_ptr<Tensor> grad = nullptr; // Store the gradient of this tensor
        std::function<void()> backward_fn = nullptr;    // Function to backpropagate the gradient
    };
}

#endif // TENSOR_H
