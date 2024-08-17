// #include "ml_framework/tensor.h"
// #include <cuda_runtime.h>

#include "tensor.h"

extern cudaError_t elementWiseMultiplyWrapper(const float *d_a, const float *d_b, float *d_c, int n);
extern float elapsedTime;

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

namespace ml_framework
{

    // Initialize the cuBLAS handle, Temporarily Global
    cublasHandle_t Tensor::cublas_handle = nullptr;

    static std::string getCuBlasErrorString(cublasStatus_t status)
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

    static int prod_shape(const std::vector<int> &shape)
    {
        int total_elements = 1;
        for (int num : shape)
        {
            total_elements *= num;
        }
        return total_elements;
    }

    std::ostream &operator<<(std::ostream &os, const Tensor &tensor)
    {
        std::vector<int> v_size = tensor.m_shape;
        float *data = tensor.h_data;

        for (int i = 0; i < tensor.data_size; ++i)
        {

            if (((i + 1) % (v_size[0] * v_size[1])) == 0)
            {
                os << *(data + i) << '\n'
                   << '\n';
            }
            else if (((i + 1) % v_size[1]) == 0)
            {
                os << *(data + i) << '\n';
            }
            else
            {
                os << *(data + i) << " ";
            }
        }
        return os;
    }
    Tensor::Tensor(const std::vector<int> &shape, const float *data)
        : m_shape(shape)
    {
        initializeCuBLAS();
        data_size = prod_shape(m_shape);
        this->h_data = new float[data_size];
        std::memcpy(this->h_data, data, sizeof(*data) * static_cast<long unsigned int>(data_size));
        // Initialize CUDA resources
        allocateDeviceMemory();
    }

    Tensor::Tensor(const Tensor &tensor)
        : m_shape(tensor.shape())
    {
        initializeCuBLAS();
        std::cout << "Copy CONSTRUCTOR" << std::endl;
        data_size = prod_shape(m_shape);
        h_data = new float[data_size];
        std::memcpy(this->h_data, tensor.h_data, sizeof(*h_data) * static_cast<long unsigned int>(data_size));

        // Initialize CUDA resources
        allocateDeviceMemory();
    }

    Tensor::Tensor(const std::vector<int> &shape)
        : m_shape(shape)
    {
        initializeCuBLAS();
        data_size = prod_shape(m_shape);
        this->h_data = new float[data_size];
        std::memset(this->h_data, 0, sizeof(*h_data) * static_cast<long unsigned int>(data_size));
        // Initialize CUDA resources
        allocateDeviceMemory();
    }

    // TODO: various member functions/constructors need to be done
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

    const std::vector<int> &Tensor::shape() const
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

    Tensor Tensor::operator+(const Tensor &other) const
    {
        if (m_shape != other.m_shape)
        {
            throw std::runtime_error("Tensor m_shapes must match for addition.");
        }

        Tensor result_tensor(*this);
        const float alpha = 1.0f;

        cublasStatus_t status = cublasSaxpy(cublas_handle, this->data_size, &alpha, other.device_data(), 1, result_tensor.device_data(), 1);
        CHECK_CUBLAS_STATUS(status);
        result_tensor.transferDataToHost();
        return result_tensor;
    }

    Tensor Tensor::operator*(const Tensor &other) const
    {
        if (m_shape != other.m_shape)
        {
            throw std::runtime_error("Tensor m_shapes must match for addition.");
        }

        Tensor result_tensor = Tensor(other);
        // const int threadsPerBlock = 256;
        // const int blocksPerGrid = (static_cast<int>(other.data_size) + threadsPerBlock - 1) / threadsPerBlock;
        cudaError_t error = elementWiseMultiplyWrapper(this->d_data, other.d_data, result_tensor.d_data, static_cast<int>(other.data_size));
        std::cout << elapsedTime << std::endl;

        CHECK_CUDA_ERROR(error);
        result_tensor.transferDataToHost();

        return result_tensor;
    }

    Tensor Tensor::matmul(const Tensor &other) const
    {
        if (m_shape.size() != 2 || other.m_shape.size() != 2)
        {
            throw std::runtime_error("Matrix multiplication requires 2D tensors.");
        }
        if (m_shape[1] != other.m_shape[0])
        {
            throw std::runtime_error("Matrix dimensions must align for multiplication.");
        }

        const std::vector<int> result_shape{m_shape[0], other.m_shape[1]};
        Tensor result_matrix(result_shape);

        // Perform matrix multiplication using cuBLAS
        const float alpha = 1.0f;
        const float beta = 0.0f;
        int m = m_shape[0];       // A rows
        int k = m_shape[1];       // A columns
        int n = other.m_shape[1]; // B columns

        cublasSetPointerMode(cublas_handle, CUBLAS_POINTER_MODE_HOST);
        cublasStatus_t status = cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha,
                                            other.d_data, n, d_data, k, &beta, result_matrix.d_data, n);
        CHECK_CUBLAS_STATUS(status);
        result_matrix.transferDataToHost();
        return result_matrix;
    }

    bool Tensor::operator==(const Tensor &other)
    {
        for (int i = 0; i < other.data_size; i++)
        {
            if (this->h_data[i] != other.h_data[i])
                return false;
        }
        return true;
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
            cublasStatus_t status = cublasCreate(&cublas_handle);
            CHECK_CUBLAS_STATUS(status);
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

} // namespace ml_framework
