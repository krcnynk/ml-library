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

namespace ml_framework
{
    class Tensor
    {
    public:
        // Constructor to initialize tensor with shape and data
        Tensor() = default;

        Tensor(const std::vector<int> &shape, const float *data);
        Tensor(const Tensor &tensor);
        Tensor(const std::vector<int> &shape);

        // Tensor();
        // Tensor &operator=(const Tensor &other);
        // Tensor(Tensor &&other) noexcept;
        // Tensor &operator=(Tensor &&other) noexcept;

        // Getter for shape
        const std::vector<int> &shape() const;
        int size() const;

        float *host_data() const;
        float *host_data();

        float *device_data() const;
        float *device_data();

        // Overload + operator
        Tensor operator+(const Tensor &other) const;
        Tensor operator-(const Tensor &other) const;
        Tensor operator*(const Tensor &other) const;
        Tensor transpose() const;
        friend Tensor operator*(float scalar, const Tensor &tensor);
        Tensor matmul(const Tensor &other) const;
        ~Tensor();
        Tensor &operator=(const Tensor &other);

        float sum() const;
        bool operator==(const Tensor &other) const;
        void transferDataToDevice() const;
        void transferDataToHost() const;

    
        friend std::ostream &operator<<(std::ostream &os, const Tensor &point);
        static void initializeCuBLAS();
        static void cleanupCuBLAS();

    private:
        std::vector<int> m_shape; // Shape of the tensor (e.g., dimensions)
        int data_size = 0;
        float *h_data = nullptr; // Data storage (flattened)

        // CUDA resources
        mutable float *d_data = nullptr;     // Device pointer for data
        static cublasHandle_t cublas_handle; // cuBLAS handle
        // Helper functions
        void allocateDeviceMemory() const;
        void freeDeviceMemory() const;
    };
}

#endif // TENSOR_H
