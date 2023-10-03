// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "ark_kernels.h"
// CAUTION: len should be even.

template <typename T>
__device__ void simple_add(T *c, T *a, T *b, unsigned int bs,
                           unsigned int len) {
    for (unsigned int i = 0; i < bs; ++i) {
        for (unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
             tid < len; tid += blockDim.x * gridDim.x) {
            c[i * len + tid] = a[i * len + tid] + b[tid];
        }
    }
}

extern "C" __global__ void simple_add_fp32(float *c, float *a, float *b,
                                           unsigned int bs, unsigned int len) {
    simple_add<float>(c, a, b, bs, len);
}

extern "C" __global__ void simple_add_fp16(ark::half *c, ark::half *a,
                                           ark::half *b, unsigned int bs,
                                           unsigned int len) {
    simple_add<__half2>((__half2 *)c, (__half2 *)a, (__half2 *)b, bs, len / 2);
}
