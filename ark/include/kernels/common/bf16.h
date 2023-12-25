// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef ARK_KERNELS_BF16_H_
#define ARK_KERNELS_BF16_H_

#include "arch.h"
#include "device.h"
#include "vector_type.h"

#if defined(ARK_TARGET_CUDA_ARCH)
#include <cuda_bf16.h>
#elif defined(ARK_TARGET_ROCM_ARCH)
#include <hip/hip_bf16.h>
#endif

namespace ark {

ARCH_ALIAS_TYPE(bf16, __nv_bfloat16, __hip_bfloat16);
ARCH_ALIAS_TYPE(bf16x2, __nv_bfloat162, __hip_bfloat162);
ARCH_ALIAS_TYPE(bf16_raw, __nv_bfloat16_raw, __hip_bfloat16);
ARCH_ALIAS_TYPE(bf16x2_raw, __nv_bfloat162_raw, __hip_bfloat162);

namespace type {

template <typename DataType>
struct Constant;

template <>
struct Constant<bf16> {
    static DEVICE bf16 zero() { return bf16_raw{0x0}; }
    static DEVICE bf16 lowest() { return bf16_raw{0xff7f}; }
};

template <>
struct Constant<bf16x2> {
    static DEVICE bf16x2 zero() { return bf16x2_raw{0x0, 0x0}; }
    static DEVICE bf16x2 lowest() { return bf16x2_raw{0xff7f, 0xff7f}; }
};

template <>
struct Vtype<bf16, 2> {
    using type = bf16x2;
};

template <>
struct Vtype<const bf16, 2> {
    using type = const bf16x2;
};

}  // namespace type

}  // namespace ark

#endif  // ARK_KERNELS_BF16_H_
