// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef ARK_KERNELS_FP16_H_
#define ARK_KERNELS_FP16_H_

#include <cuda_fp16.h>

#include "device.h"
#include "vector_type.h"

namespace ark {

using fp16 = __half;
using fp16x2 = __half2;

namespace type {

template <typename DataType>
struct Constant;

template <>
struct Constant<fp16> {
    static DEVICE fp16 zero() { return __half_raw{0}; }
    static DEVICE fp16 lowest() { return __half_raw{0xfbff}; }
};

template <>
struct Constant<fp16x2> {
    static DEVICE fp16x2 zero() { return __half2_raw{0, 0}; }
    static DEVICE fp16x2 lowest() { return __half2_raw{0xfbff, 0xfbff}; }
};

template <>
struct Vtype<fp16, 2> {
    using type = fp16x2;
};

}  // namespace type

}  // namespace ark

#endif  // ARK_KERNELS_FP16_H_
