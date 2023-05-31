// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef ARK_KERNELS_ACTIVATION_H_
#define ARK_KERNELS_ACTIVATION_H_

#include "broadcast.h"
#include "ewise.h"
#include "transform.h"

namespace ark {
template <typename InDims, typename OutDims> struct Gelu
{
    template <int NelemPerThread>
    static DEVICE void compute(__half *out, __half *in, int idx_n, int idx_c,
                               int idx_h, int idx_w)
    {
        out += idx_n * OutDims::C * OutDims::H * OutDims::W +
               idx_c * OutDims::H * OutDims::W + idx_h * OutDims::W + idx_w;
        //
        in += idx_h * InDims::C * InDims::H * InDims::W +
              idx_n * InDims::H * InDims::W + idx_c * InDims::W + idx_w;
        __half input = *in;
        __half half_pi =
            __float2half(0.7978845608f); // sqrt(2 / pi) = 0.7978845608
        __half coeff = __float2half(0.044715f);
        __half one = __float2half(1.0f);
        __half x_cubed = __hmul(input, __hmul(input, input));
        __half tanh_input = __hadd(__hmul(input, half_pi),
                                   __hmul(x_cubed, __hmul(coeff, half_pi)));
        // Convert __half to float
        float input_float = __half2float(tanh_input);

        // Compute tanh for the float variable
        float output_float = tanhf(input_float);

        // Convert float back to __half
        __half tanh_output = __float2half(output_float);

        __half gelu =
            __hmul(__hmul(input, __hadd(one, tanh_output)), __float2half(0.5f));

        *out = gelu;
    }
};

template <typename InDims, typename OutDims, typename OutShape,
          typename UnitOutShape, int ThreadsNum, int SmemBytes>
DEVICE void gelu(ark::half *out, ark::half *in, int tx, int ty, int tz)
{
    constexpr int NelemPerThread = 1;
    Ewise1<InDims, OutDims, OutShape, UnitOutShape, ThreadsNum, SmemBytes,
           Gelu<InDims, OutDims>, __half, NelemPerThread>::run((__half *)out,
                                                               (__half *)in,
                                                               tz / OutShape::C,
                                                               tz % OutShape::C,
                                                               tx, ty);
}

} // namespace ark

#endif // ARK_KERNELS_ACTIVATION_H_
