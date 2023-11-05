// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef ARK_KERNELS_BF16_H_
#define ARK_KERNELS_BF16_H_

#include <cuda_bf16.h>

#include "arch.h"
#include "device.h"
#include "vector_type.h"

namespace ark {

using bf16 = __nv_bfloat16;
using bf16x2 = __nv_bfloat162;
using bf16_raw = __nv_bfloat16_raw;

namespace type {

template <typename DataType>
struct Constant;

template <>
struct Constant<bf16> {
    static DEVICE bf16 zero() { return bf16_raw{0x0}; }
    static DEVICE bf16 lowest() { return bf16_raw{0xff7f}; }
};

template <>
struct Vtype<bf16, 2> {
    using type = bf16x2;
};

}  // namespace type

}  // namespace ark

#endif  // ARK_KERNELS_BF16_H_
