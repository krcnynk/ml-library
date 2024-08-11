// #include "ml_framework/tensor.h"
#include <cuda_runtime.h>
#include <stdexcept>
#include <iostream>
#include <memory>
#include "tensor.h"
#include <algorithm>

// if ([] {
//     static bool is_first_time = true;
//     auto was_first_time = is_first_time;
//     is_first_time = false;
//     return was_first_time; } ())
// {
//     // do the initialization part
// }

// #define FIRST_TIME_HERE ([] { \
//     static bool is_first_time = true; \
//     auto was_first_time = is_first_time; \
//     is_first_time = false; \
//     return was_first_time; }())

#define CHECK_CUDA_ERROR(err)                          \
    do                                                 \
    {                                                  \
        cudaError_t err_code = (err);                  \
        if (err_code != cudaSuccess)                   \
        {                                              \
            std::string error_msg = "CUDA error: ";    \
            error_msg += cudaGetErrorString(err_code); \
            throw std::runtime_error(error_msg);       \
        }                                              \
    } while (0)

#define CHECK_CUBLAS_STATUS(status)                    \
    do                                                 \
    {                                                  \
        cublasStatus_t err = (cublasStatus_t)(status); \
        if (err != CUBLAS_STATUS_SUCCESS)              \
        {                                              \
            std::string error_msg = "cuBLAS error: ";  \
            error_msg += getCuBlasErrorString(err);    \
            throw std::runtime_error(error_msg);       \
        }                                              \
    } while (0)

namespace ml_framework
{

    // Initialize the cuBLAS handle, Temporarily Global
    cublasHandle_t Tensor::cublas_handle = nullptr;

    static inline std::string getCuBlasErrorString(cublasStatus_t status)
    {
        switch (status)
        {
        case CUBLAS_STATUS_SUCCESS:
            return "CUBLAS_STATUS_SUCCESS";
        case CUBLAS_STATUS_NOT_INITIALIZED:
            return "CUBLAS_STATUS_NOT_INITIALIZED";
        case CUBLAS_STATUS_ALLOC_FAILED:
            return "CUBLAS_STATUS_ALLOC_FAILED";
        case CUBLAS_STATUS_INVALID_VALUE:
            return "CUBLAS_STATUS_INVALID_VALUE";
        case CUBLAS_STATUS_ARCH_MISMATCH:
            return "CUBLAS_STATUS_ARCH_MISMATCH";
        case CUBLAS_STATUS_MAPPING_ERROR:
            return "CUBLAS_STATUS_MAPPING_ERROR";
        case CUBLAS_STATUS_EXECUTION_FAILED:
            return "CUBLAS_STATUS_EXECUTION_FAILED";
        case CUBLAS_STATUS_INTERNAL_ERROR:
            return "CUBLAS_STATUS_INTERNAL_ERROR";
        case CUBLAS_STATUS_NOT_SUPPORTED:
            return "CUBLAS_STATUS_NOT_SUPPORTED";
        case CUBLAS_STATUS_LICENSE_ERROR:
            return "CUBLAS_STATUS_LICENSE_ERROR";
        default:
            return "Unknown cuBLAS status";
        }
    }

    static inline size_t prod_shape(const std::vector<size_t> &shape)
    {
        size_t total_elements = 1;
        for (size_t num : shape)
        {
            total_elements *= num;
        }
        return total_elements;
    }

    Tensor::Tensor(const std::vector<size_t> &shape, const float *data)
        : m_shape(shape)
    {
        initializeCuBLAS();
        data_size = prod_shape(m_shape);
        this->h_data = new float[data_size];
        std::memcpy(this->h_data, data, sizeof(*data) * data_size);

        // Initialize CUDA resources
        allocateDeviceMemory();
    }

    Tensor::Tensor(const Tensor &tensor)
        : m_shape(tensor.shape())
    {
        std::cout << "Copy CONSTRUCTOR" << std::endl;
        data_size = prod_shape(m_shape);
        h_data = new float[data_size];
        std::memcpy(this->h_data, tensor.host_data(), sizeof(*h_data) * data_size);

        // Initialize CUDA resources
        allocateDeviceMemory();
    }
    ///////////////////////////////////////////////////
    // Tensor::Tensor()
    // {
    //     return NULL;
    // }
    // Tensor &Tensor::operator=(const Tensor &other)
    // {
    //     std::cout << "COPY ASSIGNMENT" << std::endl;
    //     return Tensor();
    // }
    // Tensor::Tensor(Tensor &&other) noexcept
    // {
    //     std::cout << "MOVE COnstructor" << std::endl;
    //     return Tensor();
    // }

    // Tensor &Tensor::operator=(Tensor &&other) noexcept
    // {
    //     std::cout << "MOVE ASSIGNMENT" << std::endl;
    //     return Tensor();
    // }
    //////////////////////////////////////////////////////////////////////////////
    const std::vector<size_t> &Tensor::shape() const
    {
        return m_shape;
    }

    float *Tensor::host_data() const
    {
        return h_data;
    }

    float *Tensor::host_data()
    {
        return h_data;
    }

    float *Tensor::device_data() const
    {
        return d_data;
    }

    float *Tensor::device_data()
    {
        return d_data;
    }

    Tensor Tensor::operator+(const Tensor &other)
    {
        if (m_shape != other.m_shape)
        {
            throw std::runtime_error("Tensor m_shapes must match for addition.");
        }

        Tensor result_tensor(*this);
        const float alpha = 1.0f;
        cublasSaxpy(cublas_handle, this->data_size, &alpha, this->device_data(), 1, result_tensor.device_data(), 1);
        result_tensor.transferDataToHost();
        // std::memcpy(result_tensor.host_data(), result_tensor.host_data(), sizeof(float) * data_size);
        return result_tensor;
    }

    // Tensor Tensor::operator*(const Tensor &other)
    // {
    //     if (m_shape != other.m_shape)
    //     {
    //         throw std::runtime_error("Tensor m_shapes must match for multiplication.");
    //     }
    //     return Tensor(m_shape, h_data.cwiseProduct(other.h_data));
    // }

    // Tensor Tensor::matmul(const Tensor &other)
    // {
    //     if (m_shape.size() != 2 || other.m_shape.size() != 2)
    //     {
    //         throw std::runtime_error("Matrix multiplication requires 2D tensors.");
    //     }
    //     if (m_shape[1] != other.m_shape[0])
    //     {
    //         throw std::runtime_error("Matrix dimensions must align for multiplication.");
    //     }

    //     std::vector<size_t> resultm_shape(2);
    //     resultm_shape << m_shape[0], other.m_shape[1];
    //     Tensor result(resultm_shape);

    //     // Perform matrix multiplication using cuBLAS
    //     const float alpha = 1.0f;
    //     const float beta = 0.0f;
    //     int m = m_shape[0];
    //     int k = m_shape[1];
    //     int n = other.m_shape[1];

    //     // Device pointers
    //     float *d_otherh_data = nullptr;
    //     float *d_result = nullptr;

    //     cudaMalloc(&d_otherh_data, static_cast<unsigned long>(other.data_size) * sizeof(float));
    //     cudaMalloc(&d_result, static_cast<unsigned long>(result.data_size) * sizeof(float));

    //     cublasSetPointerMode(cublas_handle, CUBLAS_POINTER_MODE_HOST);
    //     cudaMemcpy(d_otherh_data, other.h_data, static_cast<unsigned long>(other.data_size) * sizeof(float), cudaMemcpyHostToDevice);

    //     cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha,
    //                 d_data, m, d_otherh_data, k, &beta, d_result, m);

    //     cudaMemcpy(result.h_data, d_result, static_cast<unsigned long>(result.data_size) * sizeof(float), cudaMemcpyDeviceToHost);

    //     cudaFree(d_otherh_data);
    //     cudaFree(d_result);

    //     return result;
    // }

    void Tensor::allocateDeviceMemory() const
    {
        if (d_data == nullptr)
        {
            cudaMalloc((void **)&d_data, static_cast<unsigned long>(data_size) * sizeof(float));
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
            cudaMemcpy(d_data, h_data, static_cast<unsigned long>(data_size) * sizeof(float), cudaMemcpyHostToDevice);

            cudaError_t err = cudaGetLastError();
            CHECK_CUDA_ERROR(err);
        }
    }

    void Tensor::transferDataToHost() const
    {
        if (d_data != nullptr)
        {
            cudaMemcpy(h_data, d_data, static_cast<unsigned long>(data_size) * sizeof(float), cudaMemcpyDeviceToHost);

            cudaError_t err = cudaGetLastError();
            CHECK_CUDA_ERROR(err);
        }
    }

    Tensor::~Tensor()
    {
        freeDeviceMemory();
        cleanupCuBLAS();
        delete[] h_data;
        h_data = nullptr;
    }

    void Tensor::initializeCuBLAS()
    {
        if (cublas_handle == nullptr)
        {
            int status = cublasCreate(&cublas_handle);
            CHECK_CUBLAS_STATUS(status);
            // if (status != CUBLAS_STATUS_SUCCESS)
            // {
            //     // std::cerr << "cublasCreate failed!" << std::endl;
            //     std::string error_msg = "cublasCreate failed!";
            //     throw std::runtime_error(error_msg);
            // }
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

} // namespace ml_framework
