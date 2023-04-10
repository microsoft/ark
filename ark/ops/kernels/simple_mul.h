// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.
#include "ark_kernels.h"
// CAUTION: len should be even.

template <typename T>
__device__ void simple_mul(T *c, T *a, T *b, unsigned int bs, unsigned int len)
{
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    while (tid < len) {
        for (unsigned int i = 0; i < bs; ++i) {
            for (unsigned int j = 0; j < len; ++j) {
                c[i * len + j] = a[i * len + j] * b[j];
            }
        }
        tid += gridDim.x * blockDim.x;
    }
}

extern "C" __global__ void simple_mul_fp32(float *c, float *a, float *b,
                                           unsigned int bs, unsigned int len)
{
    simple_mul<float>(c, a, b, bs, len);
}

extern "C" __global__ void simple_mul_fp16(ark::half *c, ark::half *a,
                                           ark::half *b, unsigned int bs,
                                           unsigned int len)
{
    simple_mul<__half2>((__half2 *)c, (__half2 *)a, (__half2 *)b, bs, len / 2);
}
