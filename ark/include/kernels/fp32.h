// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef ARK_KERNELS_FP32_H_
#define ARK_KERNELS_FP32_H_

#include <cfloat>

#include "device.h"

namespace ark {

using fp32 = float;
using fp32x2 = float2;
using fp32x4 = float4;

namespace type {

template <typename DataType>
struct Constant;

template <>
struct Constant<fp32> {
    static DEVICE fp32 zero() { return 0; }
    static DEVICE fp32 lowest() { return -FLT_MAX; }
};

}  // namespace type

}  // namespace ark

#endif  // ARK_KERNELS_FP32_H_
