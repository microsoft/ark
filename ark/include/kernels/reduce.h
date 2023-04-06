#ifndef ARK_KERNELS_REDUCE_H_
#define ARK_KERNELS_REDUCE_H_

#include "transform.h"

namespace ark {

template <bool IsRelu> struct ReduceActivation
{
};

template <> struct ReduceActivation<true>
{
    static DEVICE __half2 compute(__half2 val)
    {
#if __CUDA_ARCH__ >= 800
        return __hmax2(val, (__half2_raw){0, 0});
#else
        float2 fval = __half22float2(val);
        return __floats2half2_rn(fmaxf(fval.x, 0.0f), fmaxf(fval.y, 0.0f));
#endif
    }
};

template <> struct ReduceActivation<false>
{
    static DEVICE __half2 compute(__half2 val)
    {
        return val;
    }
};

template <int M, int N, int K, bool IsRelu, int TN, int SB, int TDM, int TDN,
          int TDK>
struct TransformReduceBatch
{
    static const int MN = M * N;

    static DEVICE __half2 compute(__half2 *x, int midx, int nidx)
    {
        __half *px = &((__half *)x)[midx + nidx * M];
        __half2 sum = *(__half2 *)px;
#pragma unroll
        for (int k = 1; k < K; ++k) {
            sum = __hadd2(sum, *(__half2 *)&px[k * MN]);
        }
        return ReduceActivation<IsRelu>::compute(sum);
    }
};

template <int M, int N, int K, bool IsRelu, int TN, int SB, int TDM, int TDN,
          int TDK>
DEVICE void reduce_batch(ark::half *y, ark::half *x, int tx, int ty, int tz)
{
    using TransformReduceBatch =
        TransformReduceBatch<M, N, K, IsRelu, TN, SB, TDM, TDN, TDK>;
    Transform<TransformReduceBatch, M, N, -1, TN, SB, TDM, TDN, TDK>::run(
        y, x, tx, ty, tz);
}

} // namespace ark

#endif // ARK_KERNELS_REDUCE_H_
