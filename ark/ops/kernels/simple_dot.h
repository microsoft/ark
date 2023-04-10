// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "ark_kernels.h"
// CAUTION: len should be even.
extern "C" __global__ void simple_dot(float *c, ark::half *a, ark::half *b,
                                      unsigned int len)
{
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid == 0) {
        unsigned int num = len / 2;
        if (num == 0) {
            *c = 0;
            return;
        }
        // __half2 *pa = (__half2 *)a;
        // __half2 *pb = (__half2 *)b;
        // __half2 sum = __hmul2(*pa, *pb);
        // for (unsigned int i = 1; i < num; ++i) {
        //     sum = __hadd2(sum, __hmul2(pa[i], pb[i]));
        // }
        // *c = __half2float(__hadd(__low2half(sum), __high2half(sum)));
        __half2 *pa = (__half2 *)a;
        __half2 *pb = (__half2 *)b;
        __half sum = 0;
        for (unsigned int i = 0; i < num; ++i) {
            __half2 tmp = __hmul2(pa[i], pb[i]);
            // printf("%f\n", __half2float(sum));
            sum = __hadd(sum, __low2half(tmp));
            // printf("%f\n", __half2float(sum));
            sum = __hadd(sum, __high2half(tmp));
        }
        // printf("%f\n", __half2float(sum));
        *c = __half2float(sum);
    }
}
