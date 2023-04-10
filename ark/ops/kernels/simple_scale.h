// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "ark_kernels.h"
// CAUTION: len should be even.
extern "C" __global__ void simple_scale(ark::half *y, ark::half *x, float val,
                                        unsigned int len)
{
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int num = len / 2;
    __half2 val2 = __float2half2_rn(val);
    __half2 *px = (__half2 *)x;
    __half2 *py = (__half2 *)y;
    while (tid < num) {
        py[tid] = __hmul2(px[tid], val2);
        tid += gridDim.x * blockDim.x;
    }
}
