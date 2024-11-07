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
    class Tensor
    {
    public:
        Tensor();
        Tensor(const std::vector<int> &shape);
        Tensor(const std::vector<int> &shape, const float *data);
        Tensor(const std::vector<int> &shape, float init);
        Tensor(const Tensor &tensor);
        // Tensor(Tensor &&tensor) noexcept;
        ~Tensor();

        const std::vector<int> &shape() const;
        size_t size() const;
        // int size() const;
        float *host_data() const;
        float *host_data();
        float *device_data() const;
        float *device_data();

        std::unique_ptr<Tensor> operator+(const Tensor &other) const;
        std::unique_ptr<Tensor> operator-(const Tensor &other) const;
        std::unique_ptr<Tensor> operator*(const Tensor &other) const;
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
        std::unique_ptr<float[]> h_data;
        mutable float *d_data;               // Device pointer for data
        static cublasHandle_t cublas_handle; // cuBLAS handle
        // Helper functions
        void allocateDeviceMemory() const;
        void freeDeviceMemory() const;

    public:
        void backward();
        mutable std::unique_ptr<Tensor> grad;
        // mutable std::shared_ptr<Tensor> grad = nullptr; // Store the gradient of this tensor
        mutable std::function<void()> backward_fn; // Function to backpropagate the gradient
        friend class Operation;
        friend class AddOperation;
        friend class MultiplyOperation;
        friend class SubOperation;
    };
}

#endif // TENSOR_H
