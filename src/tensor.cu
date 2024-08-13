
float elapsedTime;

#define tpb 256
//FIXME: This was written to time the kernel, may need some work
#define LAUNCH_KERNEL(kernel, blocksPerGrid, threadsPerBlock, d_a, d_b, d_c, n)      \
    cudaEvent_t start, stop;                                                         \
    cudaEventCreate(&start);                                                         \
    cudaEventCreate(&stop);                                                          \
    cudaEventRecord(start, 0);                                                       \
    elementWiseMultiplyKernel<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, n); \
    cudaEventRecord(stop, 0);                                                        \
    cudaEventSynchronize(stop);                                                      \
    cudaEventElapsedTime(&elapsedTime, start, stop);                                 \
    cudaEventDestroy(start);                                                         \
    cudaEventDestroy(stop);

__global__ void elementWiseMultiplyKernel(const float *d_a, const float *d_b, float *d_c, int n)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n)
    {
        d_c[idx] = d_a[idx] * d_b[idx];
    }
}

cudaError_t elementWiseMultiplyWrapper(const float *d_a, const float *d_b, float *d_c, int n)
{

    const int threadsPerBlock = tpb;
    const int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

    // Launch the kernel
    // elementWiseMultiplyKernel<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, n);
    LAUNCH_KERNEL(elementWiseMultiplyKernel, blocksPerGrid, threadsPerBlock, d_a, d_b, d_c, n);
    // Check for errors in kernel launch
    cudaError_t error = cudaGetLastError();

    return error;
}