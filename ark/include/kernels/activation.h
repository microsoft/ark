// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef ARK_KERNELS_ACTIVATION_H_
#define ARK_KERNELS_ACTIVATION_H_

#include "broadcast.h"
#include "type_intrinsics.h"

namespace ark {

struct Relu {
    template <typename DataType>
    static DEVICE DataType compute(DataType input) {
        return type::Max::compute(input, type::Constant<DataType>::zero());
    }
};

struct Gelu {
    static DEVICE float compute(float input) {
        return 0.5f * input *
               (1.0f + tanhf(0.7978845608f *
                             (input + 0.044715f * input * input * input)));
    }

    static DEVICE bf16 compute(bf16 input) {
        return type::Cast::compute<bf16>(
            Gelu::compute(type::Cast::compute<float>(input)));
    }

    static DEVICE fp16x2 compute(fp16x2 input) {
        fp16x2 half_pi =
            __float2half2_rn(0.7978845608f);  // sqrt(2 / pi) = 0.7978845608
        fp16x2 coeff = __float2half2_rn(0.044715f);
        fp16x2 one = __float2half2_rn(1.0f);

        fp16x2 x_cubed = __hmul2(input, __hmul2(input, input));
        fp16x2 tanh_input = __hadd2(__hmul2(input, half_pi),
                                    __hmul2(x_cubed, __hmul2(coeff, half_pi)));

        // Convert fp16x2 to float2
        float2 input_float2 = __half22float2(tanh_input);

        // Compute tanh for each float in the float2 variable
        float2 output_float2 =
            make_float2(tanhf(input_float2.x), tanhf(input_float2.y));

        // Convert float2 back to fp16x2
        fp16x2 tanh_output = __float22half2_rn(output_float2);

        return __hmul2(__hmul2(input, __hadd2(one, tanh_output)),
                       __float2half2_rn(0.5f));
    }
};

struct Sigmoid {
    template <typename DataType>
    static DEVICE DataType compute(DataType input) {
        return type::Div::compute(
            type::Cast::compute<DataType>(1.0f),
            (type::Add::compute(
                type::Cast::compute<DataType>(1.0f),
                type::Exp::compute(type::Neg::compute(input)))));
    }
    static DEVICE fp16x2 compute(fp16x2 input) {
        fp16x2 one = __float2half2_rn(1.0f);
        fp16x2 exp_neg_input = h2exp(__hneg2(input));
        fp16x2 one_plus_exp_neg_input = __hadd2(one, exp_neg_input);
        return __h2div(one, one_plus_exp_neg_input);
    }
};

template <typename InDims, typename InShape, typename OutDims,
          typename OutShape, typename UnitOutDims, int NumWarps, int SmemBytes,
          typename InDataType, typename OutDataType>
DEVICE void relu(OutDataType *out, InDataType *in, int uop_idx,
                 int smem_per_warp) {
    constexpr int NelemPerThread =
        (sizeof(OutDataType) <= 2 && UnitOutDims::W % 8 == 0)
            ? 8
            : (UnitOutDims::W % 4 == 0) ? 4 : (UnitOutDims::W % 2 == 0) ? 2 : 1;
    Broadcast1<InDims, InShape, OutDims, OutShape, UnitOutDims, NumWarps,
               SmemBytes,
               Broadcast1Intrinsic<Relu, InShape, InDataType, OutDataType,
                                   NelemPerThread>>::run(out, in, uop_idx);
}

template <typename InDims, typename InShape, typename OutDims,
          typename OutShape, typename UnitOutDims, int NumWarps, int SmemBytes,
          typename InDataType, typename OutDataType>
DEVICE void gelu(OutDataType *out, InDataType *in, int uop_idx,
                 int smem_per_warp) {
    constexpr int NelemPerThread =
        (sizeof(OutDataType) <= 2 && UnitOutDims::W % 8 == 0)
            ? 8
            : (UnitOutDims::W % 4 == 0) ? 4 : (UnitOutDims::W % 2 == 0) ? 2 : 1;
    Broadcast1<InDims, InShape, OutDims, OutShape, UnitOutDims, NumWarps,
               SmemBytes,
               Broadcast1Intrinsic<Gelu, InShape, InDataType, OutDataType,
                                   NelemPerThread>>::run(out, in, uop_idx);
}

template <typename InDims, typename InShape, typename OutDims,
          typename OutShape, typename UnitOutDims, int NumWarps, int SmemBytes,
          typename InDataType, typename OutDataType>
DEVICE void sigmoid(OutDataType *out, InDataType *in, int uop_idx,
                    int smem_per_warp) {
    constexpr int NelemPerThread =
        (sizeof(OutDataType) <= 2 && UnitOutDims::W % 8 == 0)
            ? 8
            : (UnitOutDims::W % 4 == 0) ? 4 : (UnitOutDims::W % 2 == 0) ? 2 : 1;
    Broadcast1<InDims, InShape, OutDims, OutShape, UnitOutDims, NumWarps,
               SmemBytes,
               Broadcast1Intrinsic<Sigmoid, InShape, InDataType, OutDataType,
                                   NelemPerThread>>::run(out, in, uop_idx);
}

}  // namespace ark

#endif  // ARK_KERNELS_ACTIVATION_H_
