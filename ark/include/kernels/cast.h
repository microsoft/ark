// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef ARK_KERNELS_CAST_H_
#define ARK_KERNELS_CAST_H_

#include "broadcast.h"

namespace ark {

template <typename _InShape, typename _FromType, typename _ToType,
          int _NelemPerThread>
struct Cast;

template <typename _InShape, typename _FromType, typename _ToType>
struct Cast<_InShape, _FromType, _ToType, 2> {
    using InputType = _FromType;
    using OutputType = _ToType;
    static const int NelemPerThread = 2;

    static DEVICE void compute(_ToType *output, const _FromType *input) {
        if constexpr (_InShape::W == 1) {
            *output = _ToType(*input);
        } else {
            output[0] = _ToType(input[0]);
            output[1] = _ToType(input[1]);
        }
    }
};

template <typename _InShape>
struct Cast<_InShape, half, float, 2> {
    using InputType = half;
    using OutputType = float;
    static const int NelemPerThread = 2;

    static DEVICE void compute(float *output, const half *input) {
        if constexpr (_InShape::W == 1) {
            *output = __half2float(*(const __half *)input);
        } else {
            float2 *pout = (float2 *)output;
            __half2 *pin = (__half2 *)input;
            *pout = __half22float2(*pin);
        }
    }
};

template <typename _InShape>
struct Cast<_InShape, int, float, 2> {
    using InputType = int;
    using OutputType = float;
    static const int NelemPerThread = 2;

    static DEVICE void compute(float *output, const int *input) {
        if constexpr (_InShape::W == 1) {
            *output = float(*input);
        } else {
            float2 *pout = (float2 *)output;
            int2 *pin = (int2 *)input;
            pout->x = float(pin->x);
            pout->y = float(pin->y);
        }
    }
};

template <typename _InShape>
struct Cast<_InShape, float, half, 2> {
    using InputType = float;
    using OutputType = half;
    static const int NelemPerThread = 2;

    static DEVICE void compute(half *output, const float *input) {
        if constexpr (_InShape::W == 1) {
            *output = __float2half_rn(*input);
        } else {
            __half2 *pout = (__half2 *)output;
            float2 *pin = (float2 *)input;
            *pout = __float22half2_rn(*pin);
        }
    }
};

template <typename _InShape>
struct Cast<_InShape, int, half, 2> {
    using InputType = int;
    using OutputType = half;
    static const int NelemPerThread = 2;

    static DEVICE void compute(half *output, const int *input) {
        if constexpr (_InShape::W == 1) {
            *output = __int2half_rn(*input);
        } else {
            __half2 *pout = (__half2 *)output;
            int2 *pin = (int2 *)input;
            *pout =
                __halves2half2(__int2half_rn(pin->x), __int2half_rn(pin->y));
        }
    }
};

template <typename _InShape>
struct Cast<_InShape, float, int, 2> {
    using InputType = float;
    using OutputType = int;
    static const int NelemPerThread = 2;

    static DEVICE void compute(int *output, const float *input) {
        if constexpr (_InShape::W == 1) {
            *output = int(*input);
        } else {
            int2 *pout = (int2 *)output;
            float2 *pin = (float2 *)input;
            pout->x = int(pin->x);
            pout->y = int(pin->y);
        }
    }
};

template <typename _InShape>
struct Cast<_InShape, half, int, 2> {
    using InputType = half;
    using OutputType = int;
    static const int NelemPerThread = 2;

    static DEVICE void compute(int *output, const half *input) {
        if constexpr (_InShape::W == 1) {
            *output = __half2int_rn(*(const __half *)input);
        } else {
            int2 *pout = (int2 *)output;
            __half2 *pin = (__half2 *)input;
            pout->x = __half2int_rn(__low2half(*pin));
            pout->y = __half2int_rn(__high2half(*pin));
        }
    }
};

// TODO: specialization for bfloat16

template <typename InDims, typename InShape, typename OutDims,
          typename OutShape, typename UnitOutDims, int NumThreads,
          typename FromType, typename ToType>
DEVICE void cast(ToType *out, FromType *in, int uop_idx, int) {
    Broadcast1<InDims, InShape, OutDims, OutShape, UnitOutDims, NumThreads, 0,
               Cast<InShape, FromType, ToType, 2>>::run(out, in, uop_idx);
}

}  // namespace ark

#endif  // ARK_KERNELS_CAST_H_
