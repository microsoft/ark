// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef ARK_KERNELS_TYPE_INTRINSICS_H_
#define ARK_KERNELS_TYPE_INTRINSICS_H_

#include "bf16.h"
#include "device.h"
#include "fp16.h"
#include "fp32.h"

namespace ark {
namespace type {

// TODO: add __nv_bfloat162 support

struct Add {
    template <typename DataType>
    static DEVICE DataType compute(DataType a, DataType b) {
        return a + b;
    }

    static DEVICE fp16x2 compute(fp16x2 a, fp16x2 b) { return __hadd2(a, b); }
};

struct Sub {
    template <typename DataType>
    static DEVICE DataType compute(DataType a, DataType b) {
        return a - b;
    }

    static DEVICE fp16x2 compute(fp16x2 a, fp16x2 b) { return __hsub2(a, b); }
};

struct Mul {
    template <typename DataType>
    static DEVICE DataType compute(DataType a, DataType b) {
        return a * b;
    }

    static DEVICE fp16x2 compute(fp16x2 a, fp16x2 b) { return __hmul2(a, b); }
};

struct Div {
    template <typename DataType>
    static DEVICE DataType compute(DataType a, DataType b) {
        return a / b;
    }

    static DEVICE fp16x2 compute(fp16x2 a, fp16x2 b) { return __h2div(a, b); }
};

struct Exp {
    template <typename DataType>
    static DEVICE DataType compute(DataType input) {
        return DataType(expf(float(input)));
    }

    static DEVICE float compute(float input) { return expf(input); }

    static DEVICE fp16x2 compute(fp16x2 input) { return h2exp(input); }
};

struct Sqrt {
    template <typename DataType>
    static DEVICE DataType compute(DataType input) {
        return DataType(sqrtf(float(input)));
    }

    static DEVICE float compute(float input) { return sqrtf(input); }

    static DEVICE fp16x2 compute(fp16x2 input) { return h2sqrt(input); }
};

struct Max {
    template <typename DataType>
    static DEVICE DataType compute(DataType a, DataType b) {
        return (a > b) ? a : b;
    }
    static DEVICE float compute(float a, float b) { return max(a, b); }
    static DEVICE fp16x2 compute(fp16x2 a, fp16x2 b) {
#if defined(ARK_TARGET_CUDA_ARCH) && (ARK_TARGET_CUDA_ARCH >= 800)
        return __hmax2(a, b);
#else
        return __halves2half2((a.x > b.x) ? a.x : b.x, (a.y > b.y) ? a.y : b.y);
#endif  // (ARK_TARGET_CUDA_ARCH >= 800)
    }
};

}  // namespace type
}  // namespace ark

#endif  // ARK_KERNELS_TYPE_INTRINSICS_H_
