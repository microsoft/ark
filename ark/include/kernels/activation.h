// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef ARK_KERNELS_ACTIVATION_H_
#define ARK_KERNELS_ACTIVATION_H_

#include "broadcast.h"
#include "ewise.h"
#include "transform.h"

namespace ark {

template <int M> struct TransformGELU
{
    static __device__ __half2 compute(__half2 *a, int midx, int nidx)
    {
        __half2 input = *(__half2 *)&((__half *)a)[midx + nidx * M];
        __half2 half_pi =
            __float2half2_rn(0.7978845608f); // sqrt(2 / pi) = 0.7978845608
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

        __half2 gelu = __hmul2(__hmul2(input, __hadd2(one, tanh_output)),
                               __float2half2_rn(0.5f));
        return gelu;
    }
};

template <int M, int N, int TN, int SB, int TDM, int TDN, int TDK>
DEVICE void gelu(ark::half *y, ark::half *x, int tx, int ty, int tz)
{
    Transform<TransformGELU<M>, M, N, 1, TN, SB, TDM, TDN, TDK>::run(y, x, tx,
                                                                     ty, tz);
}

} // namespace ark

#endif // ARK_KERNELS_ACTIVATION_H_
