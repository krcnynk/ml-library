#include "ml_framework/tensor.h"
#include <cuda_runtime.h>
#include <stdexcept>
#include <iostream>

namespace ml_framework
{

    // Initialize the cuBLAS handle
    cublasHandle_t Tensor::cublas_handle = nullptr;

    
    Tensor::Tensor(const Eigen::VectorXi &shape)
        : _shape(shape)
    {
        int total_elements = shape.prod();
        _data.resize(total_elements);

        // Initialize CUDA resources
        allocateDeviceMemory();
    }

    Tensor::Tensor(const Eigen::VectorXi &shape, const Eigen::VectorXf &data)
        : _shape(shape), _data(data)
    {
        int total_elements = shape.prod();
        if (data.size() != total_elements)
        {
            throw std::runtime_error("Data size does not match shape.");
        }

        // Initialize CUDA resources
        allocateDeviceMemory();
        transferDataToDevice();
    }

    const Eigen::VectorXi &Tensor::shape() const
    {
        return _shape;
    }

    const Eigen::VectorXf &Tensor::data() const
    {
        return _data;
    }

    Eigen::VectorXf &Tensor::data()
    {
        return _data;
    }

    Tensor Tensor::operator+(const Tensor &other) const
    {
        if (_shape != other._shape)
        {
            throw std::runtime_error("Tensor shapes must match for addition.");
        }
        return Tensor(_shape, _data + other._data);
    }

    Tensor Tensor::operator*(const Tensor &other) const
    {
        if (_shape != other._shape)
        {
            throw std::runtime_error("Tensor shapes must match for multiplication.");
        }
        return Tensor(_shape, _data.cwiseProduct(other._data));
    }

    Tensor Tensor::matmul(const Tensor &other) const
    {
        if (_shape.size() != 2 || other._shape.size() != 2)
        {
            throw std::runtime_error("Matrix multiplication requires 2D tensors.");
        }
        if (_shape[1] != other._shape[0])
        {
            throw std::runtime_error("Matrix dimensions must align for multiplication.");
        }

        Eigen::VectorXi result_shape(2);
        result_shape << _shape[0], other._shape[1];
        Tensor result(result_shape);

        // Perform matrix multiplication using cuBLAS
        const float alpha = 1.0f;
        const float beta = 0.0f;
        int m = _shape[0];
        int k = _shape[1];
        int n = other._shape[1];

        // Device pointers
        float *d_other_data = nullptr;
        float *d_result = nullptr;

        cudaMalloc(&d_other_data, other._data.size() * sizeof(float));
        cudaMalloc(&d_result, result._data.size() * sizeof(float));

        cublasSetPointerMode(cublas_handle, CUBLAS_POINTER_MODE_HOST);
        cudaMemcpy(d_other_data, other._data.data(), other._data.size() * sizeof(float), cudaMemcpyHostToDevice);

        cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha,
                    d_data, m, d_other_data, k, &beta, d_result, m);

        cudaMemcpy(result._data.data(), d_result, result._data.size() * sizeof(float), cudaMemcpyDeviceToHost);

        cudaFree(d_other_data);
        cudaFree(d_result);

        return result;
    }

    void Tensor::allocateDeviceMemory() const
    {
        if (d_data == nullptr)
        {
            cudaMalloc((void**)&d_data, _data.size() * sizeof(float));
            transferDataToDevice();
        }
    }

    void Tensor::freeDeviceMemory() const
    {
        if (d_data != nullptr)
        {
            cudaFree(d_data);
            d_data = nullptr;
        }
    }

    void Tensor::transferDataToDevice() const
    {
        if (d_data != nullptr)
        {
            cudaMemcpy(d_data, _data.data(), _data.size() * sizeof(float), cudaMemcpyHostToDevice);
        }
    }

    void Tensor::transferDataToHost() const
    {
        if (d_data != nullptr)
        {
            cudaMemcpy(d_data, d_data, _data.size() * sizeof(float), cudaMemcpyDeviceToHost);
        }
    }

    void Tensor::initializeCuBLAS()
    {
        if (cublas_handle == nullptr)
        {
            cublasCreate(&cublas_handle);
        }
    }

    void Tensor::cleanupCuBLAS()
    {
        if (cublas_handle != nullptr)
        {
            cublasDestroy(cublas_handle);
            cublas_handle = nullptr;
        }
    }

    Tensor::~Tensor()
    {
        freeDeviceMemory();
    }

} // namespace ml_framework
