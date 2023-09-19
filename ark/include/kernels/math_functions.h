// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef ARK_KERNELS_MATH_FUNCTIONS_H_
#define ARK_KERNELS_MATH_FUNCTIONS_H_

#include "broadcast.h"

namespace ark {

struct Exp
{
    static DEVICE float compute(const float &input)
    {
        return expf(input);
    }
    static DEVICE __half compute(const __half &input)
    {
        return hexp(input);
    }
    static DEVICE __half2 compute(const __half2 &input)
    {
        return h2exp(input);
    }
};

struct Sqrt
{
    static DEVICE float compute(const float &input)
    {
        return sqrtf(input);
    }
    static DEVICE __half compute(const __half &input)
    {
        return hsqrt(input);
    }
    static DEVICE __half2 compute(const __half2 &input)
    {
        return h2sqrt(input);
    }
};

struct Rsqrt
{
    static DEVICE float compute(const float &input)
    {
        return rsqrtf(input);
    }
    static DEVICE __half compute(const __half &input)
    {
        return hrsqrt(input);
    }
    static DEVICE __half2 compute(const __half2 &input)
    {
        return h2rsqrt(input);
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

template <typename _MathType, typename _InShape>
struct Math<_MathType, _InShape, float, 1>
{
    using DataType = float;
    static const int NelemPerThread = 1;

    static DEVICE void compute(float *output, const float *input)
    {
        *output = _MathType::compute(*input);
    }
};

template <typename InDims, typename InShape, typename OutDims,
          typename OutShape, typename UnitOutDims, int NumThreads,
          int SmemBytes>
DEVICE void exp(half *out, half *in, int uop_idx, int)
{
    Broadcast1<InDims, InShape, OutDims, OutShape, UnitOutDims, NumThreads,
               SmemBytes, Math<Exp, InShape, half, 2>>::run(out, in, uop_idx);
}

template <typename InDims, typename InShape, typename OutDims,
          typename OutShape, typename UnitOutDims, int NumThreads,
          int SmemBytes>
DEVICE void exp(float *out, float *in, int uop_idx, int)
{
    Broadcast1<InDims, InShape, OutDims, OutShape, UnitOutDims, NumThreads,
               SmemBytes, Math<Exp, InShape, float, 1>>::run(out, in, uop_idx);
}

template <typename InDims, typename InShape, typename OutDims,
          typename OutShape, typename UnitOutDims, int NumThreads,
          int SmemBytes>
DEVICE void sqrt(half *out, half *in, int uop_idx, int)
{
    Broadcast1<InDims, InShape, OutDims, OutShape, UnitOutDims, NumThreads,
               SmemBytes, Math<Sqrt, InShape, half, 2>>::run(out, in, uop_idx);
}

template <typename InDims, typename InShape, typename OutDims,
          typename OutShape, typename UnitOutDims, int NumThreads,
          int SmemBytes>
DEVICE void sqrt(float *out, float *in, int uop_idx, int)
{
    Broadcast1<InDims, InShape, OutDims, OutShape, UnitOutDims, NumThreads,
               SmemBytes, Math<Sqrt, InShape, float, 1>>::run(out, in, uop_idx);
}

} // namespace ark

#endif // ARK_KERNELS_MATH_FUNCTIONS_H_
