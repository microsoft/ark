// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "ark_kernels.h"
// CAUTION: n*m should be even.
extern "C" __global__ void simple_reduce(ark::half *y, ark::half *x,
                                         unsigned int m, unsigned int n,
                                         unsigned int k, bool is_relu) {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int num = n * m / 2;
    while (tid < num) {
        __half2 *py = (__half2 *)y;
        __half2 *px = (__half2 *)x;
        __half2 sum = px[tid];
        for (unsigned int i = 1; i < k; ++i) {
            sum = __hadd2(sum, px[tid + i * num]);
        }
        if (is_relu) {
            float2 fsum = __half22float2(sum);
            py[tid] =
                __floats2half2_rn(fmaxf(fsum.x, 0.0f), fmaxf(fsum.y, 0.0f));
        } else {
            py[tid] = sum;
        }
        tid += gridDim.x * blockDim.x;
    }
}
