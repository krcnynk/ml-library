#ifndef TENSOR_H
#define TENSOR_H

#include "suppress_warnings.h"
SUPPRESS_SIGN_CONVERSION_WARNINGS
RESTORE_WARNINGS

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <vector>

// using custom_type = float;

namespace ml_framework
{
    class Tensor
    {
    public:
        // Constructor to initialize tensor with shape and data
        Tensor(const std::vector<size_t> &shape, const float *data);

        Tensor(const Tensor &tensor);

        // Tensor();
        // Tensor &operator=(const Tensor &other);
        // Tensor(Tensor &&other) noexcept;
        // Tensor &operator=(Tensor &&other) noexcept;

        // Getter for shape
        const std::vector<size_t> &shape() const;

        float *host_data() const;
        float *host_data();

        float *device_data() const;
        float *device_data();

        // Overload + operator
        Tensor operator+(const Tensor &other);

        // Overload * operator (element-wise multiplication)
        Tensor operator*(const Tensor &other);

        // Matrix multiplication using cuBLAS
        Tensor matmul(const Tensor &other);

        // Cleanup CUDA resources
        ~Tensor();

        void elementWiseMultiply(const float* d_a, const float* d_b, float* d_c, int n);
        void transferDataToDevice() const;
        void transferDataToHost() const;
        static void initializeCuBLAS();
        static void cleanupCuBLAS();

    private:
        std::vector<size_t> m_shape; // Shape of the tensor (e.g., dimensions)
        size_t data_size = 0;
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
