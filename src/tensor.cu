#include "tensor.h"

__global__ void elementWiseMultiplyKernel(const float* d_a, const float* d_b, float* d_c, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        d_c[idx] = d_a[idx] * d_b[idx];
    }
}