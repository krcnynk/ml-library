// config.h
#ifndef CONFIG_H
#define CONFIG_H

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <string>

// #ifdef __GNUC__
// #define ML_CACHE_ALIGN __attribute__((aligned(64)))
// #elif defined(_MSC_VER)
// #define ML_CACHE_ALIGN __declspec(align(64))
// #else
// #define ML_CACHE_ALIGN
// #endif

extern float elapsedTime;
extern cudaError_t elementWiseMultiplyKernelWrapper(const float *d_a, const float *d_b, float *d_c, int n);
extern cudaError_t reluKernelWrapper(float *d_input, float *d_output, int n);
extern cudaError_t leakyReluKernelWrapper(float *d_input, float *d_output, float gradient, int n);
extern cudaError_t d_leakyReluKernelWrapper(float *d_input, float *d_output, float gradient, int n);

// extern __global__ void leakyReluKernel();

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
        if (status != CUBLAS_STATUS_SUCCESS)           \
        {                                              \
            std::string error_msg = "cuBLAS error: ";  \
            error_msg += getCuBlasErrorString(status); \
            throw std::runtime_error(error_msg);       \
        }                                              \
    } while (0)

inline std::string getCuBlasErrorString(cublasStatus_t status)
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

inline void printVector(const std::vector<int> &vec)
{
    for (const auto &element : vec)
    {
        std::cout << element << " ";
    }
    std::cout << std::endl;
}

#endif // CONFIG_H
