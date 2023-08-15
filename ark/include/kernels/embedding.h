// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef ARK_KERNELS_EMBEDDING_H_
#define ARK_KERNELS_EMBEDDING_H_

#include "arithmetic.h"

namespace ark {

struct RoPE
{
    using DataType = float;
    static const int NelemPerThread = 2;
    // RoPE: https://arxiv.org/pdf/2104.09864.pdf
    static DEVICE void compute(float *c, const float *a, const float *b)
    {
        float2 *pc = (float2 *)c;
        const float2 *pa = (const float2 *)a;
        const float2 *pb = (const float2 *)b;
        printf("pa->x: %f, pa->y: %f pb->x: %f, pb->y: %f\n", pa->x, pa->y,
               pb->x, pb->y);
        pc->x = pa->x * pb->x - pa->y * pb->y;
        pc->y = pa->x * pb->y + pa->y * pb->x;
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
               UnitOutDims, NumThreads, SmemBytes, RoPE>::run(c, a, b, uop_idx);
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
