// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef ARK_KERNELS_FP16_H_
#define ARK_KERNELS_FP16_H_

#if defined(ARK_TARGET_CUDA_ARCH)
#include <cuda_fp16.h>
#elif defined(ARK_TARGET_ROCM_ARCH)
#include <hip/hip_fp16.h>
#endif

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
    static DEVICE fp16 zero() {
#if defined(ARK_TARGET_CUDA_ARCH)
        return __half_raw{0};
#elif defined(ARK_TARGET_ROCM_ARCH)
        union BitCast {
            unsigned short u;
            fp16 f;
        };
        return BitCast{0}.f;
#endif
    }
    static DEVICE fp16 lowest() {
#if defined(ARK_TARGET_CUDA_ARCH)
        return __half_raw{0xfbff};
#elif defined(ARK_TARGET_ROCM_ARCH)
        union BitCast {
            unsigned short u;
            fp16 f;
        };
        return BitCast{0xfbff}.f;
#endif
    }
};

template <>
struct Constant<fp16x2> {
    static DEVICE fp16x2 zero() {
#if defined(ARK_TARGET_CUDA_ARCH)
        return __half2_raw{0, 0};
#elif defined(ARK_TARGET_ROCM_ARCH)
        union BitCast {
            unsigned short u[2];
            fp16x2 f;
        };
        return BitCast{0, 0}.f;
#endif
    }
    static DEVICE fp16x2 lowest() {
#if defined(ARK_TARGET_CUDA_ARCH)
        return __half2_raw{0xfbff, 0xfbff};
#elif defined(ARK_TARGET_ROCM_ARCH)
        union BitCast {
            unsigned short u[2];
            fp16x2 f;
        };
        return BitCast{0xfbff, 0xfbff}.f;
#endif
    }
};

template <>
struct Vtype<fp16, 2> {
    using type = fp16x2;
};

}  // namespace type

}  // namespace ark

#endif  // ARK_KERNELS_FP16_H_
