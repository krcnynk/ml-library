#ifndef TENSOR_H
#define TENSOR_H

#include "suppress_warnings.h"
SUPPRESS_SIGN_CONVERSION_WARNINGS
#include <Eigen/Dense>
RESTORE_WARNINGS
#include <cublas_v2.h>

namespace ml_framework
{
    class Tensor
    {
    public:
        // Constructor to initialize tensor with shape
        Tensor(const Eigen::VectorXi &shape);

        // Constructor to initialize tensor with shape and data
        Tensor(const Eigen::VectorXi &shape, const Eigen::VectorXf &data);

        // Getter for shape
        const Eigen::VectorXi &shape() const;

        // Getter for data
        const Eigen::VectorXf &data() const;

        // Getter for data (non-const)
        Eigen::VectorXf &data();

        // Overload + operator
        Tensor operator+(const Tensor &other) const;

        // Overload * operator (element-wise multiplication)
        Tensor operator*(const Tensor &other) const;

        // Matrix multiplication using cuBLAS
        Tensor matmul(const Tensor &other) const;

        // Cleanup CUDA resources
        ~Tensor();

    private:
        Eigen::VectorXi _shape; // Shape of the tensor (e.g., dimensions)
        Eigen::VectorXf _data;  // Data storage (flattened)

        // CUDA resources
        mutable float *d_data = nullptr; // Device pointer for data
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
