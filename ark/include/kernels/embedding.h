// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef ARK_KERNELS_EMBEDDING_H_
#define ARK_KERNELS_EMBEDDING_H_

#include "arithmetic.h"

namespace ark {

// Rotary Position Embedding(RoPE): https://arxiv.org/pdf/2104.09864.pdf

template <typename DataType> struct RoPE;

template <> struct RoPE<float>
{
    using DataType = float;
    static const int NelemPerThread = 2;
    static DEVICE void compute(float *c, const float *a, const float *b)
    {
        float2 *pc = (float2 *)c;
        const float2 *pa = (const float2 *)a;
        const float2 *pb = (const float2 *)b;
        pc->x = pa->x * pb->x - pa->y * pb->y;
        pc->y = pa->x * pb->y + pa->y * pb->x;
    }
};

template <> struct RoPE<half>
{
    using DataType = half;
    static const int NelemPerThread = 2;
    static DEVICE void compute(half *c, const half *a, const half *b)
    {
        __half2 *pc = (__half2 *)c;
        const __half2 *pa = (const __half2 *)a;
        const __half2 *pb = (const __half2 *)b;
        pc->x = __hmul(pa->x, pb->x) - __hmul(pa->y, pb->y);
        pc->y = __hmul(pa->x, pb->y) + __hmul(pa->y, pb->x);
    }
};

template <typename In0Dims, typename In0Shape, typename In1Dims,
          typename In1Shape, typename OutDims, typename OutShape,
          typename UnitOutDims, int NumThreads, int SmemBytes>
DEVICE void rope(float *c, float *a, float *b, int uop_idx, int)
{
    Broadcast2<In0Dims, In0Shape, In1Dims, In1Shape, OutDims, OutShape,
               UnitOutDims, NumThreads, SmemBytes, RoPE<float>>::run(c, a, b,
                                                                     uop_idx);
}

template <typename In0Dims, typename In0Shape, typename In1Dims,
          typename In1Shape, typename OutDims, typename OutShape,
          typename UnitOutDims, int NumThreads, int SmemBytes>
DEVICE void rope(half *c, half *a, half *b, int uop_idx, int)
{
    Broadcast2<In0Dims, In0Shape, In1Dims, In1Shape, OutDims, OutShape,
               UnitOutDims, NumThreads, SmemBytes, RoPE<half>>::run(c, a, b,
                                                                    uop_idx);
}

} // namespace ark

#endif // ARK_KERNELS_EMBEDDING_H_
