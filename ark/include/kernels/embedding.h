// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef ARK_KERNELS_EMBEDDING_H_
#define ARK_KERNELS_EMBEDDING_H_

#include "arithmetic.h"

namespace ark {

struct RoPE
{
    using DataType = float2;
    static const int NelemPerThread = 1;
    // RoPE: https://arxiv.org/pdf/2104.09864.pdf
    static DEVICE float2 compute(float2 a, float2 b)
    {
        float2 result;
        result.x = a.x * b.x - a.y * b.y;
        result.y = a.x * b.y + a.y * b.x;
        return result;
    }
    static DEVICE __half2 compute(__half2 a, __half2 b)
    {
        return __hmul2(a, b);
    }
};

template <typename In0Dims, typename In0Shape, typename In1Dims,
          typename In1Shape, typename OutDims, typename OutShape,
          typename UnitOutDims, int NumThreads, int SmemBytes>
DEVICE void rope(float *c, float *a, float *b, int uop_idx, int)
{
    Broadcast2<In0Dims, In0Shape, In1Dims, In1Shape, OutDims, OutShape,
               UnitOutDims, NumThreads, SmemBytes, RoPE>::run((float2 *)c,
                                                              (float2 *)a,
                                                              (float2 *)b,
                                                              uop_idx);
}

// template <typename In0Dims, typename In0Shape, typename In1Dims,
//           typename In1Shape, typename OutDims, typename OutShape,
//           typename UnitOutDims, int NumThreads, int SmemBytes>
// DEVICE void rope(half *c, half *a, half *b, int uop_idx, int)
// {
//     Broadcast2<In0Dims, In0Shape, In1Dims, In1Shape, OutDims, OutShape,
//                UnitOutDims, NumThreads, SmemBytes, RoPE<half2>>::run((half2
//                *)c,
//                                                                      (half2
//                                                                      *)a,
//                                                                      (half2
//                                                                      *)b,
//                                                                      uop_idx);
// }

} // namespace ark

#endif // ARK_KERNELS_EMBEDDING_H_
