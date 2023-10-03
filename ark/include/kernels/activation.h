// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef ARK_KERNELS_ACTIVATION_H_
#define ARK_KERNELS_ACTIVATION_H_

#include "broadcast.h"

namespace ark {

struct Relu {
    static DEVICE float compute(float input) { return max(input, 0.0f); }
    static DEVICE __half2 compute(__half2 input) {
        return __hmax2(input, (__half2_raw){0, 0});
    }
};

struct Gelu {
    static DEVICE float compute(float input) {
        return 0.5f * input *
               (1.0f + tanhf(0.7978845608f *
                             (input + 0.044715f * input * input * input)));
    }

    static DEVICE __half2 compute(__half2 input) {
        __half2 half_pi =
            __float2half2_rn(0.7978845608f);  // sqrt(2 / pi) = 0.7978845608
        __half2 coeff = __float2half2_rn(0.044715f);
        __half2 one = __float2half2_rn(1.0f);

        __half2 x_cubed = __hmul2(input, __hmul2(input, input));
        __half2 tanh_input = __hadd2(__hmul2(input, half_pi),
                                     __hmul2(x_cubed, __hmul2(coeff, half_pi)));

        // Convert __half2 to float2
        float2 input_float2 = __half22float2(tanh_input);

        // Compute tanh for each float in the float2 variable
        float2 output_float2 =
            make_float2(tanhf(input_float2.x), tanhf(input_float2.y));

        // Convert float2 back to __half2
        __half2 tanh_output = __float22half2_rn(output_float2);

        return __hmul2(__hmul2(input, __hadd2(one, tanh_output)),
                       __float2half2_rn(0.5f));
    }
};

struct Sigmoid {
    static DEVICE float compute(float input) {
        return 1.0f / (1.0f + expf(-input));
    }
    static DEVICE __half2 compute(__half2 input) {
        __half2 one = __float2half2_rn(1.0f);
        __half2 exp_neg_input = h2exp(__hneg2(input));
        __half2 one_plus_exp_neg_input = __hadd2(one, exp_neg_input);
        return __h2div(one, one_plus_exp_neg_input);
    }
};

template <typename _ActivationType, typename _InShape, typename _DataType,
          int _NelemPerThread>
struct Activation;

template <typename _ActivationType, typename _InShape>
struct Activation<_ActivationType, _InShape, half, 2> {
    using InputType = half;
    using OutputType = half;
    static const int NelemPerThread = 2;

    static DEVICE void compute(half *output, const half *input) {
        __half2 *pout = (__half2 *)output;
        if (_InShape::W == 1) {
            *pout =
                _ActivationType::compute(__half2half2(*(const __half *)input));
        } else {
            __half2 *pin = (__half2 *)input;
            *pout = _ActivationType::compute(*pin);
        }
    }
};

template <typename _ActivationType, typename _InShape>
struct Activation<_ActivationType, _InShape, float, 1> {
    using InputType = float;
    using OutputType = float;
    static const int NelemPerThread = 1;

    static DEVICE void compute(float *output, const float *input) {
        *output = _ActivationType::compute(*input);
    }
};

template <typename InDims, typename InShape, typename OutDims,
          typename OutShape, typename UnitOutDims, int NumThreads,
          int SmemBytes>
DEVICE void relu(float *out, float *in, int uop_idx, int) {
    Broadcast1<InDims, InShape, OutDims, OutShape, UnitOutDims, NumThreads,
               SmemBytes, Activation<Relu, InShape, float, 1>>::run(out, in,
                                                                    uop_idx);
}

template <typename InDims, typename InShape, typename OutDims,
          typename OutShape, typename UnitOutDims, int NumThreads,
          int SmemBytes>
DEVICE void relu(half *out, half *in, int uop_idx, int) {
    Broadcast1<InDims, InShape, OutDims, OutShape, UnitOutDims, NumThreads,
               SmemBytes, Activation<Relu, InShape, half, 2>>::run(out, in,
                                                                   uop_idx);
}

template <typename InDims, typename InShape, typename OutDims,
          typename OutShape, typename UnitOutDims, int NumThreads,
          int SmemBytes>
DEVICE void gelu(float *out, float *in, int uop_idx, int) {
    Broadcast1<InDims, InShape, OutDims, OutShape, UnitOutDims, NumThreads,
               SmemBytes, Activation<Gelu, InShape, float, 1>>::run(out, in,
                                                                    uop_idx);
}

template <typename InDims, typename InShape, typename OutDims,
          typename OutShape, typename UnitOutDims, int NumThreads,
          int SmemBytes>
DEVICE void gelu(half *out, half *in, int uop_idx, int) {
    Broadcast1<InDims, InShape, OutDims, OutShape, UnitOutDims, NumThreads,
               SmemBytes, Activation<Gelu, InShape, half, 2>>::run(out, in,
                                                                   uop_idx);
}

template <typename InDims, typename InShape, typename OutDims,
          typename OutShape, typename UnitOutDims, int NumThreads,
          int SmemBytes>
DEVICE void sigmoid(float *out, float *in, int uop_idx, int) {
    Broadcast1<InDims, InShape, OutDims, OutShape, UnitOutDims, NumThreads,
               SmemBytes, Activation<Sigmoid, InShape, float, 1>>::run(out, in,
                                                                       uop_idx);
}

template <typename InDims, typename InShape, typename OutDims,
          typename OutShape, typename UnitOutDims, int NumThreads,
          int SmemBytes>
DEVICE void sigmoid(half *out, half *in, int uop_idx, int) {
    Broadcast1<InDims, InShape, OutDims, OutShape, UnitOutDims, NumThreads,
               SmemBytes, Activation<Sigmoid, InShape, half, 2>>::run(out, in,
                                                                      uop_idx);
}

}  // namespace ark

#endif  // ARK_KERNELS_ACTIVATION_H_
