// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef ARK_KERNELS_BF16_H_
#define ARK_KERNELS_BF16_H_

#if defined(ARK_TARGET_CUDA_ARCH)
#include <cuda_bf16.h>
#endif

#include "device.h"

namespace ark {

using bf16 = __nv_bfloat16;
using bf16x2 = __nv_bfloat162;

namespace type {

template <typename DataType>
struct Constant;

template <>
struct Constant<bf16> {
    static DEVICE bf16 zero() { return __nv_bfloat16_raw{0x0}; }
    static DEVICE bf16 lowest() { return __nv_bfloat16_raw{0xff7f}; }
};

}  // namespace type

}  // namespace ark

#endif  // ARK_KERNELS_BF16_H_
