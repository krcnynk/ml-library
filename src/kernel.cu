
static int tpb = 256;
float elapsedTime;

__global__ void elementWiseMultiplyKernel(const float *d_a, const float *d_b, float *d_c, int n)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n)
    {
        d_c[idx] = d_a[idx] * d_b[idx];
    }
}

__global__ void reluKernel(float *d_input, float *d_output, int n)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n)
    {
        d_output[idx] = max(0.0f, d_input[idx]);
    }
}

__global__ void leakyReluKernel(float *d_input, float *d_output, float gradient, int n)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n)
    {
        if (d_input[idx] > 0)
            d_output[idx] = d_input[idx];
        else
            d_output[idx] = gradient * d_input[idx];
    }
}

cudaError_t elementWiseMultiplyKernelWrapper(const float *d_a, const float *d_b, float *d_c, int n)
{
    // static float elapsedTime;
    const unsigned int threadsPerBlock = tpb;
    const unsigned int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

    // cudaEvent_t start, stop;
    // cudaEventCreate(&start);
    // cudaEventCreate(&stop);
    // cudaEventRecord(start, 0);

    elementWiseMultiplyKernel<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, n);
    cudaError_t error = cudaGetLastError();
    // cudaEventRecord(stop, 0);
    // cudaEventSynchronize(stop);
    // cudaEventElapsedTime(&elapsedTime, start, stop);
    // cudaEventDestroy(start);
    // cudaEventDestroy(stop);
    return error;
}

cudaError_t reluKernelWrapper(float *d_input, float *d_output, int n)
{

    const unsigned int threadsPerBlock = tpb;
    const unsigned int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

    reluKernel<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, n);
    cudaError_t error = cudaGetLastError();
    return error;
}

cudaError_t leakyReluKernelWrapper(float *d_input, float *d_output,float gradient, int n)
{

    const unsigned int threadsPerBlock = tpb;
    const unsigned int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

    leakyReluKernel<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, gradient, n);
    cudaError_t error = cudaGetLastError();
    return error;
}