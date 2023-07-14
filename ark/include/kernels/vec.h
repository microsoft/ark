// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef ARK_KERNELS_VEC_H_
#define ARK_KERNELS_VEC_H_

#include "static_math.h"

namespace ark {

using DimType = int;

template <DimType _D0 = 1, DimType _D1 = 1, DimType _D2 = 1, DimType _D3 = 1>
struct Vec
{
    static_assert(_D0 >= 0, "");
    static_assert(_D1 >= 0, "");
    static_assert(_D2 >= 0, "");
    static_assert(_D3 >= 0, "");

    // 4D representation.
    static const DimType D0 = _D0;
    static const DimType D1 = _D1;
    static const DimType D2 = _D2;
    static const DimType D3 = _D3;
    static const DimType N = _D0;
    static const DimType C = _D1;
    static const DimType H = _D2;
    static const DimType W = _D3;

    // 3D representation.
    static const DimType X = _D0;
    static const DimType Y = _D1;
    static const DimType Z = _D2;

    // Multiplied values.
    static const DimType NCHW =
        math::mul<N, math::mul<C, math::mul<H, W>::value>::value>::value;
    static const DimType NCH = math::mul<N, math::mul<C, H>::value>::value;
    static const DimType NCW = math::mul<N, math::mul<C, W>::value>::value;
    static const DimType NHW = math::mul<N, math::mul<H, W>::value>::value;
    static const DimType CHW = math::mul<C, math::mul<H, W>::value>::value;
    static const DimType NC = math::mul<N, C>::value;
    static const DimType NH = math::mul<N, H>::value;
    static const DimType NW = math::mul<N, W>::value;
    static const DimType CH = math::mul<C, H>::value;
    static const DimType CW = math::mul<C, W>::value;
    static const DimType HW = math::mul<H, W>::value;
};

} // namespace ark

#endif // ARK_KERNELS_VEC_H_
