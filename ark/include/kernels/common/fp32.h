// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef ARK_KERNELS_FP32_H_
#define ARK_KERNELS_FP32_H_

#include <cfloat>

#include "device.h"
#include "vector_type.h"

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

template <>
struct Constant<fp32x2> {
    static DEVICE fp32x2 zero() { return make_float2(0, 0); }
    static DEVICE fp32x2 lowest() { return make_float2(-FLT_MAX, -FLT_MAX); }
};

template <>
struct Constant<fp32x4> {
    static DEVICE fp32x4 zero() { return make_float4(0, 0, 0, 0); }
    static DEVICE fp32x4 lowest() {
        return make_float4(-FLT_MAX, -FLT_MAX, -FLT_MAX, -FLT_MAX);
    }
};

template <>
struct Vtype<fp32, 2> {
    using type = fp32x2;
};

template <>
struct Vtype<const fp32, 2> {
    using type = const fp32x2;
};

template <>
struct Vtype<fp32, 4> {
    using type = fp32x4;
};

template <>
struct Vtype<const fp32, 4> {
    using type = const fp32x4;
};

}  // namespace type

}  // namespace ark

#endif  // ARK_KERNELS_FP32_H_
