
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
    const int threadsPerBlock = 256;
    const int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

    // Launch the kernel
    elementWiseMultiplyKernel<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, n);

    // Check for errors in kernel launch
    cudaError_t error = cudaGetLastError();
    return error;
}