// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef ARK_KERNELS_MATH_H_
#define ARK_KERNELS_MATH_H_

#include "broadcast.h"

namespace ark {

struct Sqrt
{
    static DEVICE __half2 compute(__half2 input)
    {
        return h2sqrt(input);
    }
};

template <typename _MathType, typename _InShape, typename _DataType,
          int _NelemPerThread>
struct Math;

template <typename _MathType, typename _InShape>
struct Math<_MathType, _InShape, half, 2>
{
    using DataType = half;
    static const int NelemPerThread = 2;

    static DEVICE void compute(half *output, const half *input)
    {
        __half2 *pout = (__half2 *)output;
        if (_InShape::W == 1) {
            *pout = _MathType::compute(__half2half2(*(const __half *)input));
        } else {
            __half2 *pin = (__half2 *)input;
            *pout = _MathType::compute(*pin);
        }
    }
};

template <typename InDims, typename InShape, typename OutDims,
          typename OutShape, typename UnitOutDims, int NumThreads,
          int SmemBytes>
DEVICE void sqrt(half *out, half *in, int uop_idx, int)
{
    Broadcast1<InDims, InShape, OutDims, OutShape, UnitOutDims, NumThreads,
               SmemBytes, Math<Sqrt, InShape, half, 2>>::run(out, in, uop_idx);
}

} // namespace ark

#endif // ARK_KERNELS_MATH_H_
