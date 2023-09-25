// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef ARK_KERNELS_CAST_H_
#define ARK_KERNELS_CAST_H_

#include "broadcast.h"

namespace ark {

template <typename _InShape, typename _FromType, typename _ToType,
          int _NelemPerThread>
struct Cast;

template <typename _InShape> struct Cast<_InShape, half, float, 2>
{
    using InputType = half;
    using OutputType = float;
    static const int NelemPerThread = 2;

    static DEVICE void compute(float *output, const half *input)
    {
        if constexpr (_InShape::W == 1) {
            *output = __half2float(*(const __half *)input);
        } else {
            float2 *pout = (float2 *)output;
            __half2 *pin = (__half2 *)input;
            *pout = __half22float2(*pin);
        }
    }
};

template <typename _InShape> struct Cast<_InShape, float, half, 2>
{
    using InputType = float;
    using OutputType = half;
    static const int NelemPerThread = 2;

    static DEVICE void compute(half *output, const float *input)
    {
        if constexpr (_InShape::W == 1) {
            *output = __float2half_rn(*input);
        } else {
            __half2 *pout = (__half2 *)output;
            float2 *pin = (float2 *)input;
            *pout = __float22half2_rn(*pin);
        }
    }
};

template <typename InDims, typename InShape, typename OutDims,
          typename OutShape, typename UnitOutDims, int NumThreads>
DEVICE void cast(float *out, half *in, int uop_idx, int)
{
    Broadcast1<InDims, InShape, OutDims, OutShape, UnitOutDims, NumThreads, 0,
               Cast<InShape, half, float, 2>>::run(out, in, uop_idx);
}

template <typename InDims, typename InShape, typename OutDims,
          typename OutShape, typename UnitOutDims, int NumThreads>
DEVICE void cast(half *out, float *in, int uop_idx, int)
{
    Broadcast1<InDims, InShape, OutDims, OutShape, UnitOutDims, NumThreads, 0,
               Cast<InShape, float, half, 2>>::run(out, in, uop_idx);
}

} // namespace ark

#endif // ARK_KERNELS_CAST_H_
