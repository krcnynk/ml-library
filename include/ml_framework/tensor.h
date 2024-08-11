#ifndef TENSOR_H
#define TENSOR_H

#include "suppress_warnings.h"
SUPPRESS_SIGN_CONVERSION_WARNINGS
RESTORE_WARNINGS
#include <cublas_v2.h>
#include <vector>

using custom_type = float;

    namespace ml_framework
{
    class Tensor
    {
    public:
        // Constructor to initialize tensor with shape
        Tensor(const std::vector<size_t> &shape);

        // Constructor to initialize tensor with shape and data
        Tensor(const std::vector<size_t> &shape, const custom_type *data);

        // Getter for shape
        const std::vector<size_t> &shape() const;

        // Getter for data
        const custom_type *data() const;

        // Getter for data (non-const)
        custom_type *data();

        // Overload + operator
        Tensor operator+(const Tensor &other) const;

        // Overload * operator (element-wise multiplication)
        Tensor operator*(const Tensor &other) const;

        // Matrix multiplication using cuBLAS
        Tensor matmul(const Tensor &other) const;

        // Cleanup CUDA resources
        ~Tensor();

    private:
        std::vector<size_t> _shape; // Shape of the tensor (e.g., dimensions)
        size_t _data_size;
        custom_type *_data;              // Data storage (flattened)

        // CUDA resources
        mutable float *d_data = nullptr;     // Device pointer for data
        static cublasHandle_t cublas_handle; // cuBLAS handle

        // Helper functions
        void allocateDeviceMemory() const;
        void freeDeviceMemory() const;
        void transferDataToDevice() const;
        void transferDataToHost() const;
        static void initializeCuBLAS();
        static void cleanupCuBLAS();
    };
}

#endif // TENSOR_H
