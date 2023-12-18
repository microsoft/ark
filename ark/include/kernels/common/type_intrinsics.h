// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef ARK_KERNELS_TYPE_INTRINSICS_H_
#define ARK_KERNELS_TYPE_INTRINSICS_H_

#include <type_traits>

#include "bf16.h"
#include "device.h"
#include "fp16.h"
#include "fp32.h"
#include "integer.h"
#include "vector_type.h"

namespace ark {
namespace type {

struct Cast {
    template <typename CastType, typename DataType>
    static DEVICE CastType compute(DataType input) {
        if constexpr (std::is_same<CastType, DataType>::value) {
            return input;
        } else if constexpr (std::is_same<CastType, fp16>::value &&
                             std::is_same<DataType, float>::value) {
            return __float2half_rn(input);
        } else if constexpr (std::is_same<CastType, float>::value &&
                             std::is_same<DataType, fp16>::value) {
            return __half2float(input);
        } else if constexpr (std::is_same<CastType, fp16x2>::value &&
                             std::is_same<DataType, float2>::value) {
            return __float22half2_rn(input);
        } else if constexpr (std::is_same<CastType, float2>::value &&
                             std::is_same<DataType, fp16x2>::value) {
            return __half22float2(input);
        } else if constexpr (std::is_same<CastType, bf16>::value &&
                             std::is_same<DataType, float>::value) {
            return __float2bfloat16(input);
        } else if constexpr (std::is_same<CastType, float>::value &&
                             std::is_same<DataType, bf16>::value) {
            return __bfloat162float(input);
        } else if constexpr (std::is_same<CastType, bf16x2>::value &&
                             std::is_same<DataType, float2>::value) {
            return __float22bfloat162_rn(input);
        } else if constexpr (std::is_same<CastType, float2>::value &&
                             std::is_same<DataType, bf16x2>::value) {
            return __bfloat1622float2(input);
        } else if constexpr (std::is_same<CastType, int>::value &&
                             std::is_same<DataType, float>::value) {
            return int(input);
        } else if constexpr (std::is_same<CastType, float>::value &&
                             std::is_same<DataType, int>::value) {
            return float(input);
        } else if constexpr (std::is_same<CastType, int>::value &&
                             std::is_same<DataType, fp16>::value) {
            return __half2int_rn(input);
        } else if constexpr (std::is_same<CastType, fp16>::value &&
                             std::is_same<DataType, int>::value) {
            return __int2half_rn(input);
        } else if constexpr (std::is_same<CastType, int>::value &&
                             std::is_same<DataType, bf16>::value) {
#if defined(ARK_TARGET_CUDA_ARCH)
            return __bfloat162int_rn(input);
#elif defined(ARK_TARGET_ROCM_ARCH)
            return Cast::compute<int>(Cast::compute<float>(input));
#endif
        } else if constexpr (std::is_same<CastType, bf16>::value &&
                             std::is_same<DataType, int>::value) {
#if defined(ARK_TARGET_CUDA_ARCH)
            return __int2bfloat16_rn(input);
#elif defined(ARK_TARGET_ROCM_ARCH)
            return Cast::compute<bf16>(Cast::compute<float>(input));
#endif
        } else if constexpr (std::is_same<CastType, unsigned int>::value &&
                             std::is_same<DataType, float>::value) {
            return (unsigned int)(input);
        } else if constexpr (std::is_same<CastType, float>::value &&
                             std::is_same<DataType, unsigned int>::value) {
            return float(input);
        } else if constexpr (std::is_same<CastType, unsigned int>::value &&
                             std::is_same<DataType, fp16>::value) {
            return __half2uint_rn(input);
        } else if constexpr (std::is_same<CastType, fp16>::value &&
                             std::is_same<DataType, unsigned int>::value) {
            return __uint2half_rn(input);
        } else if constexpr (std::is_same<CastType, unsigned int>::value &&
                             std::is_same<DataType, bf16>::value) {
#if defined(ARK_TARGET_CUDA_ARCH)
            return __bfloat162uint_rn(input);
#elif defined(ARK_TARGET_ROCM_ARCH)
            return Cast::compute<unsigned int>(Cast::compute<float>(input));
#endif
        } else if constexpr (std::is_same<CastType, bf16>::value &&
                             std::is_same<DataType, unsigned int>::value) {
#if defined(ARK_TARGET_CUDA_ARCH)
            return __uint2bfloat16_rn(input);
#elif defined(ARK_TARGET_ROCM_ARCH)
            return Cast::compute<bf16>(Cast::compute<float>(input));
#endif
        } else if constexpr (std::is_same<CastType, int2>::value &&
                             std::is_same<DataType, fp16x2>::value) {
            int2 output;
            output.x = Cast::compute<int>(__low2half(input));
            output.y = Cast::compute<int>(__high2half(input));
            return output;
        } else if constexpr (std::is_same<CastType, fp16x2>::value &&
                             std::is_same<DataType, int2>::value) {
            return __halves2half2(Cast::compute<fp16>(input.x),
                                  Cast::compute<fp16>(input.y));
        } else if constexpr (std::is_same<CastType, int2>::value &&
                             std::is_same<DataType, bf16x2>::value) {
            int2 output;
            output.x = Cast::compute<int>(__low2bfloat16(input));
            output.y = Cast::compute<int>(__high2bfloat16(input));
            return output;
        } else if constexpr (std::is_same<CastType, bf16x2>::value &&
                             std::is_same<DataType, int2>::value) {
            return __halves2bfloat162(Cast::compute<bf16>(input.x),
                                      Cast::compute<bf16>(input.y));
        } else if constexpr (IsBuiltinVector2<CastType>::value &&
                             IsBuiltinVector2<DataType>::value) {
            CastType output;
            output.x = Cast::compute<decltype(output.x)>(input.x);
            output.y = Cast::compute<decltype(output.y)>(input.y);
            return output;
        } else if constexpr (IsBuiltinVector4<CastType>::value &&
                             IsBuiltinVector4<DataType>::value) {
            CastType output;
            output.x = Cast::compute<decltype(output.x)>(input.x);
            output.y = Cast::compute<decltype(output.y)>(input.y);
            output.z = Cast::compute<decltype(output.z)>(input.z);
            output.w = Cast::compute<decltype(output.w)>(input.w);
            return output;
        } else {
            return CastType(input);
        }
    }
};

struct Add {
    template <typename DataType>
    static DEVICE DataType compute(DataType a, DataType b) {
        return a + b;
    }

    static DEVICE fp16 compute(fp16 a, fp16 b) { return __hadd(a, b); }

    static DEVICE fp16x2 compute(fp16x2 a, fp16x2 b) { return __hadd2(a, b); }

    static DEVICE bf16 compute(bf16 a, bf16 b) { return __hadd(a, b); }

    static DEVICE bf16x2 compute(bf16x2 a, bf16x2 b) { return __hadd2(a, b); }
};

struct Sub {
    template <typename DataType>
    static DEVICE DataType compute(DataType a, DataType b) {
        return a - b;
    }

    static DEVICE fp16 compute(fp16 a, fp16 b) { return __hsub(a, b); }

    static DEVICE fp16x2 compute(fp16x2 a, fp16x2 b) { return __hsub2(a, b); }

    static DEVICE bf16 compute(bf16 a, bf16 b) { return __hsub(a, b); }

    static DEVICE bf16x2 compute(bf16x2 a, bf16x2 b) { return __hsub2(a, b); }
};

struct Mul {
    template <typename DataType>
    static DEVICE DataType compute(DataType a, DataType b) {
        return a * b;
    }

    static DEVICE fp16 compute(fp16 a, fp16 b) { return __hmul(a, b); }

    static DEVICE fp16x2 compute(fp16x2 a, fp16x2 b) { return __hmul2(a, b); }

    static DEVICE bf16 compute(bf16 a, bf16 b) { return __hmul(a, b); }

    static DEVICE bf16x2 compute(bf16x2 a, bf16x2 b) { return __hmul2(a, b); }
};

struct Div {
    template <typename DataType>
    static DEVICE DataType compute(DataType a, DataType b) {
        return a / b;
    }

    static DEVICE fp16 compute(fp16 a, fp16 b) { return __hdiv(a, b); }

    static DEVICE fp16x2 compute(fp16x2 a, fp16x2 b) { return __h2div(a, b); }

    static DEVICE bf16 compute(bf16 a, bf16 b) { return __hdiv(a, b); }

    static DEVICE bf16x2 compute(bf16x2 a, bf16x2 b) { return __h2div(a, b); }
};

struct Neg {
    template <typename DataType>
    static DEVICE DataType compute(DataType input) {
        return -input;
    }

    static DEVICE fp16 compute(fp16 input) { return __hneg(input); }

    static DEVICE fp16x2 compute(fp16x2 input) { return __hneg2(input); }

    static DEVICE bf16 compute(bf16 input) { return __hneg(input); }

    static DEVICE bf16x2 compute(bf16x2 input) { return __hneg2(input); }
};

struct Exp {
    template <typename DataType>
    static DEVICE DataType compute(DataType input) {
        return Cast::compute<DataType>(expf(Cast::compute<float>(input)));
    }

    static DEVICE fp16 compute(fp16 input) { return hexp(input); }

    static DEVICE fp16x2 compute(fp16x2 input) { return h2exp(input); }

    static DEVICE bf16 compute(bf16 input) { return hexp(input); }

    static DEVICE bf16x2 compute(bf16x2 input) { return h2exp(input); }
};

struct Sqrt {
    template <typename DataType>
    static DEVICE DataType compute(DataType input) {
        return Cast::compute<DataType>(sqrtf(Cast::compute<float>(input)));
    }

    static DEVICE fp16 compute(fp16 input) { return hsqrt(input); }

    static DEVICE fp16x2 compute(fp16x2 input) { return h2sqrt(input); }

    static DEVICE bf16 compute(bf16 input) { return hsqrt(input); }

    static DEVICE bf16x2 compute(bf16x2 input) { return h2sqrt(input); }
};

struct Rsqrt {
    template <typename DataType>
    static DEVICE DataType compute(DataType input) {
        return Cast::compute<DataType>(rsqrtf(Cast::compute<float>(input)));
    }

    static DEVICE fp16 compute(fp16 input) { return hrsqrt(input); }

    static DEVICE fp16x2 compute(fp16x2 input) { return h2rsqrt(input); }

    static DEVICE bf16 compute(bf16 input) { return hrsqrt(input); }

    static DEVICE bf16x2 compute(bf16x2 input) { return h2rsqrt(input); }
};

struct Max {
    template <typename DataType>
    static DEVICE DataType compute(DataType a, DataType b) {
        return (a > b) ? a : b;
    }

    static DEVICE float compute(float a, float b) { return fmaxf(a, b); }

    static DEVICE fp16 compute(fp16 a, fp16 b) { return __hmax(a, b); }

    static DEVICE fp16x2 compute(fp16x2 a, fp16x2 b) {
#if defined(ARK_TARGET_CUDA_ARCH) && (ARK_TARGET_CUDA_ARCH >= 800)
        return __hmax2(a, b);
#else
        return __halves2half2(Max::compute(__low2half(a), __low2half(b)),
                              Max::compute(__high2half(a), __high2half(b)));
#endif
    }

    static DEVICE bf16 compute(bf16 a, bf16 b) { return __hmax(a, b); }

    static DEVICE bf16x2 compute(bf16x2 a, bf16x2 b) { return __hmax2(a, b); }
};

}  // namespace type
}  // namespace ark

#endif  // ARK_KERNELS_TYPE_INTRINSICS_H_
