// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef ARK_KERNELS_TYPE_INTRINSICS_H_
#define ARK_KERNELS_TYPE_INTRINSICS_H_

// #include "bfloat16.h"
// #include "half.h"
#include <cfloat>

namespace ark {
namespace type {

// TODO: add __nv_bfloat162 support

template <typename DataType>
struct Constant {
    static DEVICE DataType zero() { return DataType(0); }
    static DEVICE DataType lowest() { return -FLT_MAX; }
};

// template <>
// struct Constant<__half2> {
//     static DEVICE __half2 zero() { return __half2_raw{0, 0}; }
//     static DEVICE __half2 lowest() { return __half2_raw{0xfbff, 0xfbff}; }
// };

// template <>
// struct Constant<bfloat16> {
//     static DEVICE bfloat16 zero() { return bfloat16(0); }
//     static DEVICE bfloat16 lowest() { return bfloat16::bitcast(0xff7f); }
// };

struct Add {
    template <typename DataType>
    static DEVICE DataType compute(DataType a, DataType b) {
        return a + b;
    }
    // static DEVICE __half2 compute(__half2 a, __half2 b) {
    //     return __hadd2(a, b);
    // }
    // struct DEVICE __nv_bfloat162 compute(__nv_bfloat162 a, __nv_bfloat162 b)
    // {
    //     return __hadd2(a, b);
    // }
};

struct Sub {
    template <typename DataType>
    static DEVICE DataType compute(DataType a, DataType b) {
        return a - b;
    }
    // static DEVICE __half2 compute(__half2 a, __half2 b) {
    //     return __hsub2(a, b);
    // }
    // struct DEVICE __nv_bfloat162 compute(__nv_bfloat162 a, __nv_bfloat162 b)
    // {
    //     return __hsub2(a, b);
    // }
};

struct Mul {
    template <typename DataType>
    static DEVICE DataType compute(DataType a, DataType b) {
        return a * b;
    }
    // static DEVICE __half2 compute(__half2 a, __half2 b) {
    //     return __hmul2(a, b);
    // }
    // struct DEVICE __nv_bfloat162 compute(__nv_bfloat162 a, __nv_bfloat162 b)
    // {
    //     return __hmul2(a, b);
    // }
};

struct Div {
    template <typename DataType>
    static DEVICE DataType compute(DataType a, DataType b) {
        return a / b;
    }
    // static DEVICE __half2 compute(__half2 a, __half2 b) {
    //     return __h2div(a, b);
    // }
    // struct DEVICE __nv_bfloat162 compute(__nv_bfloat162 a, __nv_bfloat162 b)
    // {
    //     return __h2div(a, b);
    // }
};

struct Exp {
    static DEVICE float compute(float input) { return expf(input); }
    // static DEVICE half compute(half input) { return half(expf(float(input)));
    // } static DEVICE bfloat16 compute(bfloat16 input) {
    //     return bfloat16(expf(float(input)));
    // }
    // static DEVICE __half2 compute(__half2 input) { return h2exp(input); }
    // struct DEVICE __nv_bfloat162 compute(__nv_bfloat162 input) {
    //     return h2exp(input);
    // }
};

struct Sqrt {
    static DEVICE float compute(float input) { return sqrtf(input); }
    // static DEVICE half compute(half input) { return half(sqrtf(float(input))); }
    // static DEVICE bfloat16 compute(bfloat16
    // input) {
    //     return bfloat16(sqrtf(float(input)));
    // }
    // static DEVICE __half2 compute(__half2 input) { return h2sqrt(input); }
    // struct DEVICE __nv_bfloat162 compute(__nv_bfloat162 input) {
    //     return h2sqrt(input);
    // }
};

struct Max {
    template <typename DataType>
    static DEVICE DataType compute(DataType a, DataType b) {
        return (a > b) ? a : b;
    }
    static DEVICE float compute(float a, float b) { return max(a, b); }
    //     static DEVICE __half2 compute(__half2 a, __half2 b) {
    // #if (__CUDA_ARCH__ >= 800)
    //         return __hmax2(a, b);
    // #else
    //         return __halves2half2((a.x > b.x) ? a.x : b.x, (a.y > b.y) ? a.y
    //         : b.y);
    // #endif  // (__CUDA_ARCH__ >= 800)
    //     }
};

}  // namespace type
}  // namespace ark

#endif  // ARK_KERNELS_TYPE_INTRINSICS_H_
