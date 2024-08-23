// #include "ml_framework/tensor.h"
// #include <cuda_runtime.h>
#include "autograd.h"
#include "tensor.h"
// extern cudaError_t elementWiseMultiplyWrapper(const float *d_a, const float *d_b, float *d_c, int n);
// extern float elapsedTime;

namespace ml_framework
{
    // Initialize the cuBLAS handle, Temporarily Global
    cublasHandle_t Tensor::cublas_handle = nullptr;

    static size_t prod_shape(const std::vector<int> &shape)
    {
        int total_elements = 1;
        for (int num : shape)
        {
            total_elements *= num;
        }
        return static_cast<size_t>(total_elements);
    }

    std::ostream &operator<<(std::ostream &os, const Tensor &tensor)
    {
        std::vector<size_t> v_size{tensor.size()};
        std::transform(tensor.m_shape.begin(), tensor.m_shape.end(), v_size.begin(), [](int value)
                       { return static_cast<size_t>(value); });
        float *data = tensor.h_data.get();
        std::cout << std::fixed << std::setprecision(2);
        for (size_t i = 0; i < tensor.size(); ++i)
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

    Tensor::Tensor() : m_shape(0), h_data(nullptr), d_data(nullptr), grad(nullptr) {}

    Tensor::Tensor(const std::vector<int> &shape)
        : m_shape(shape), h_data(std::make_unique<float[]>(size())), d_data(std::make_unique<float[]>(size())), grad(std::make_shared<Tensor>())
    {
        initializeCuBLAS();
        // FIXME: Need recap here
        std::random_device rd;
        // unsigned int seed = 1234;
        std::mt19937 gen(rd()); // Seed the generator with rd()
        std::normal_distribution<float> dist(0.0, std::sqrt(2.0f / static_cast<float>(size())));
        // std::uniform_real_distribution<> dis(-0.001, 0.001);

        std::fill_n(h_data.get(), size(), 0.0f);
        // std::generate(h_data.get(), h_data.get() + size(), [&]()
        //               { return dist(gen); });
        allocateDeviceMemory();

        grad->m_shape = shape;
        grad->h_data = std::make_unique<float[]>(size());
        grad->grad = nullptr;
        std::fill_n(grad->h_data.get(), size(), 0.0f);
        grad->allocateDeviceMemory();
    }

    Tensor::Tensor(const std::vector<int> &shape, float init)
        : m_shape(shape), h_data(std::make_unique<float[]>(size())), d_data(std::make_unique<float[]>(size())), grad(std::make_shared<Tensor>(shape))
    {
        initializeCuBLAS();
        std::fill_n(h_data.get(), size(), init);
        allocateDeviceMemory();
    }

    Tensor::Tensor(const std::vector<int> &shape, const float *data)
        : m_shape(shape), h_data(std::make_unique<float[]>(size())), d_data(std::make_unique<float[]>(size())), grad(std::make_shared<Tensor>(shape))
    {
        initializeCuBLAS();
        std::memcpy(this->h_data.get(), data, sizeof(*data) * static_cast<long unsigned int>(size()));
        allocateDeviceMemory();
    }

    // Tensor::Tensor(const Tensor &tensor)
    //     : m_shape(tensor.m_shape), h_data(std::make_unique<float[]>(tensor.size())), d_data(std::make_unique<float[]>(tensor.size())),
    //       grad(std::make_shared<Tensor>(tensor.m_shape, tensor.grad->h_data.get())), backward_fn(tensor.backward_fn)
    // {
    //     initializeCuBLAS();
    //     std::memcpy(this->h_data.get(), tensor.h_data.get(), sizeof(*(h_data.get())) * static_cast<long unsigned int>(size()));
    //     allocateDeviceMemory();
    // }

    // Move Constructor
    Tensor::Tensor(Tensor &&other) noexcept
        : m_shape(std::move(other.m_shape)),
          h_data(std::move(other.h_data)),
          d_data(std::move(other.d_data)),
          grad(std::move(other.grad)),
          backward_fn(std::move(other.backward_fn))
    {
        // Transfer ownership
    }

    // TODO: various member functions/constructors need to be done
    ///////////////////////////////////////////////////
    // Tensor::Tensor()
    // {
    //     retur n NULL;
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
    size_t Tensor::size() const
    {
        return prod_shape(m_shape);
    }

    const std::vector<int> &Tensor::shape() const
    {
        return m_shape;
    }

    float *Tensor::host_data() const
    {
        return h_data.get();
    }

    float *Tensor::host_data()
    {
        return h_data.get();
    }

    float *Tensor::device_data() const
    {
        return d_data.get();
    }

    float *Tensor::device_data()
    {
        return d_data.get();
    }

    float Tensor::sum() const
    {
        if (!h_data.get())
        {
            throw std::runtime_error("Host data is nullptr.");
        }
        float sum = 0;
        for (size_t i = 0; i < size(); i++)
        {
            sum = sum + h_data.get()[i];
        }
        return sum;
    }

    Tensor Tensor::operator+(const Tensor &other) const
    {
        // Create the AddOperation
        auto add_operation = std::make_shared<AddOperation>();
        // Forward pass
        Tensor result_tensor = add_operation->forward(*this, other);
        
        // Define the backward function
        // result_tensor.backward_fn = [this, &other, &result_tensor, add_operation]()
        // {
        //     // Call backward to get gradients
        //     std::vector<Tensor> grads = add_operation->backward(result_tensor.grad);

        //     // Accumulate gradients
        //     if (this->grad == nullptr || other.grad == nullptr)
        //     {
        //         throw std::runtime_error("Gradient not initialized before accumulation.");
        //     }
        //     *this->grad = add_operation->forward(*this->grad, grads[0]);
        //     *other.grad = add_operation->forward(*other.grad, grads[1]);
        // };

        return result_tensor;
    }

    // Tensor Tensor::operator-(const Tensor &other) const
    // {
    //     if (m_shape != other.m_shape)
    //     {
    //         throw std::runtime_error("Tensor m_shapes must match for addition.");
    //     }

    //     Tensor result_tensor(this->m_shape,this->h_data);
    //     const float alpha = -1.0f;

    //     cublasStatus_t status = cublasSaxpy(cublas_handle, static_cast<int>(this->size()), &alpha, other.device_data(), 1, result_tensor.device_data(), 1);
    //     CHECK_CUBLAS_STATUS(status);
    //     result_tensor.transferDataToHost();

    //     // // backward function for autograd defined here
    //     // result_tensor.backward_fn = [this, &other, &result_tensor]()
    //     // {
    //     //     if (this->grad == nullptr)
    //     //         this->grad = std::make_shared<Tensor>(this->m_shape, 0.0f);
    //     //     if (other.grad == nullptr)
    //     //         other.grad = std::make_shared<Tensor>(other.m_shape, 0.0f);
    //     //     // if (result.grad == nullptr)
    //     //     //     result.grad = std::make_shared<Tensor>(result.m_shape, 0.0f);

    //     //     *this->grad = *this->grad + *result_tensor.grad; // Gradient for this tensor
    //     //     *other.grad = *other.grad - *result_tensor.grad; // Gradient for other tensor
    //     // };

    //     return result_tensor;
    // }

    // Tensor Tensor::operator*(const Tensor &other) const
    // {
    //     if (m_shape != other.m_shape)
    //     {
    //         throw std::runtime_error("Tensor m_shapes must match for addition.");
    //     }

    //     Tensor result_tensor = Tensor(other.m_shape,other.h_data);
    //     cudaError_t err = elementWiseMultiplyKernelWrapper(this->d_data.get(), other.d_data.get(), result_tensor.d_data.get(), static_cast<int>(this->size()));
    //     CHECK_CUDA_ERROR(err);
    //     result_tensor.transferDataToHost();

    //     // result_tensor.backward_fn = [this, &other, &result_tensor]()
    //     // {
    //     //     if (this->grad == nullptr)
    //     //         this->grad = std::make_shared<Tensor>(this->m_shape, 0.0f);
    //     //     if (other.grad == nullptr)
    //     //         other.grad = std::make_shared<Tensor>(other.m_shape, 0.0f);

    //     //     // Propagate gradients: dL/da = dL/dc * b and dL/db = dL/dc * a
    //     //     *this->grad += *result_tensor.grad * other;
    //     //     *other.grad += *result_tensor.grad * *this;
    //     // };

    //     return result_tensor;
    // }

    // Tensor operator*(float scalar, const Tensor &other)
    // {
    //     Tensor dummy_tensor = Tensor(other.m_shape,other.h_data);
    //     Tensor result_tensor = Tensor(other.m_shape,other.h_data);
    //     dummy_tensor.h_data = std::make_shared<float[]>(dummy_tensor.size());
    //     std::fill_n(dummy_tensor.h_data.get(), dummy_tensor.size(), scalar);
    //     // Launch the kernel
    //     cudaError_t err = elementWiseMultiplyKernelWrapper(dummy_tensor.d_data.get(), other.d_data.get(), result_tensor.d_data.get(), static_cast<int>(dummy_tensor.size()));
    //     CHECK_CUDA_ERROR(err);
    //     result_tensor.transferDataToHost();

    //     // result_tensor.backward_fn = [&scalar, &other, &result_tensor]()
    //     // {
    //     //     if (other.grad == nullptr)
    //     //         other.grad = std::make_shared<Tensor>(other.m_shape, 0.0f);

    //     //     // Propagate gradients: dL/da = dL/dc * b and dL/db = dL/dc * a
    //     //     *other.grad += scalar * *result_tensor.grad;
    //     // };

    //     return result_tensor;
    // }

    // Tensor Tensor::transpose() const
    // {
    //     std::vector<int> transposed_shape = {this->m_shape[1], this->m_shape[0]};
    //     Tensor return_tensor = Tensor(transposed_shape, this->h_data.get());

    //     for (int i = 0; i < return_tensor.m_shape[0]; ++i)
    //     {
    //         for (int j = 0; j < return_tensor.m_shape[1]; ++j)
    //         {
    //             // Transpose operation: swap rows and columns
    //             return_tensor.h_data.get()[j * return_tensor.m_shape[0] + i] = this->h_data.get()[i * return_tensor.m_shape[1] + j];
    //         }
    //     }

    //     return_tensor.transferDataToDevice();
    //     return return_tensor;
    // }

    // Tensor Tensor::matmul(const Tensor &other) const
    // {

    //     if (m_shape.size() != 2 || other.m_shape.size() != 2)
    //     {
    //         throw std::runtime_error("Matrix multiplication requires 2D tensors.");
    //     }
    //     if (m_shape[1] != other.m_shape[0])
    //     {
    //         // printVector(this->shape());
    //         // printVector(other.shape());
    //         throw std::runtime_error("Matrix dimensions must align for multiplication.");
    //     }

    //     const std::vector<int> result_shape{m_shape[0], other.m_shape[1]};
    //     Tensor result_matrix(result_shape);

    //     // Perform matrix multiplication using cuBLAS
    //     const float alpha = 1.0f;
    //     const float beta = 0.0f;
    //     int m = m_shape[0];       // A rows
    //     int k = m_shape[1];       // A columns
    //     int n = other.m_shape[1]; // B columns 5 3

    //     cublasSetPointerMode(cublas_handle, CUBLAS_POINTER_MODE_HOST);
    //     cublasStatus_t status = cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha,
    //                                         other.d_data.get(), n, d_data.get(), k, &beta, result_matrix.d_data.get(), n);
    //     CHECK_CUBLAS_STATUS(status);
    //     result_matrix.transferDataToHost();
    //     return result_matrix;
    // }

    Tensor &Tensor::operator=(Tensor &&other) noexcept
    {
        std::cout << "move assignment op" << std::endl;
        if (this != &other)
        {
            // if (this != &tensor)
            // {
            // Free existing resources
            // freeDeviceMemory();

            // Move resources from tensor
            m_shape = std::move(other.m_shape);
            h_data = std::move(other.h_data);
            d_data = std::move(other.d_data);
            grad = std::move(other.grad);
            backward_fn = std::move(other.backward_fn);

            // Transfer ownership
        }
        return *this;
    }

    bool Tensor::operator==(const Tensor &other) const
    {
        for (size_t i = 0; i < other.size(); i++)
        {
            if (this->h_data.get()[i] != other.h_data.get()[i])
                return false;
        }
        return true;
    }

    void Tensor::transferDataToDevice() const
    {
        if (d_data != nullptr)
        {
            cudaMemcpy(d_data.get(), h_data.get(), static_cast<unsigned long>(size()) * sizeof(float), cudaMemcpyHostToDevice);

            cudaError_t err = cudaGetLastError();
            CHECK_CUDA_ERROR(err);
        }
    }

    void Tensor::transferDataToHost() const
    {
        if (d_data != nullptr)
        {
            cudaMemcpy(h_data.get(), d_data.get(), static_cast<unsigned long>(size()) * sizeof(float), cudaMemcpyDeviceToHost);

            cudaError_t err = cudaGetLastError();
            CHECK_CUDA_ERROR(err);
        }
    }

    void Tensor::backward()
    {
        if (backward_fn)
        {
            backward_fn(); // Trigger backpropagation
        }
    }

    Tensor::~Tensor()
    {
        freeDeviceMemory();
        cleanupCuBLAS();
        // delete[] h_data;
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
            float *raw_ptr = d_data.get();
            cudaMalloc((void **)&raw_ptr, static_cast<unsigned long>(size()) * sizeof(float));
            transferDataToDevice();
        }
    }

    void Tensor::freeDeviceMemory() const
    {
        if (d_data != nullptr)
        {
            cudaFree(d_data.get());
            d_data = nullptr;
        }
    }

    // void Tensor::backward()
    // {
    //     if (grad == nullptr)
    //     {
    //         grad = std::make_shared<float[]><Tensor>(shape, 1.0f)); // Initialize gradient to 1 for starting point
    //     }
    //     if (operation)
    //     {
    //         std::vector<std::shared_ptr<Tensor>> grad_inputs = operation->backward(grad);
    //         for (auto &input : grad_inputs)
    //         {
    //             input->backward(); // Recursively call backward on inputs
    //         }
    //     }
    // }

} // namespace ml_framework
