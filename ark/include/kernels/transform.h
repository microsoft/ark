// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef ARK_KERNELS_TRANSFORM_H_
#define ARK_KERNELS_TRANSFORM_H_

#include "base_op.h"
#include "static_math.h"

namespace ark {

//
template <typename TType, int LDM, int LDN, int BcastType, int TN, int SB,
          int TDM, int TDN, int TDK>
struct Transform
{
    using BaseOp = BaseOp<LDM, TN, SB, TDM, TDN, TDK>;

    //
    static DEVICE void run(ark::half *y, ark::half *x, int t0, int t1, int t2)
    {
        static_assert(LDM % TDM == 0, "");

        constexpr int NNumPerLoop = math::max<1, 2 * TN / TDM>::value;
        constexpr int MNumPerLoop = math::min<TDM, 2 * TN>::value;

        constexpr int IterNDiff = NNumPerLoop * LDM;
        constexpr int IterMNum = math::div_up<TDM, MNumPerLoop>::value;
        constexpr int IterNNum = TDN / NNumPerLoop;

        int midx = BaseOp::midx(t0, 2 * BaseOp::thread_id());
        int nidx = BaseOp::nidx(t1, 2 * BaseOp::thread_id());
#pragma unroll
        for (int i = 0; i < IterNNum; ++i) {
#pragma unroll
            for (int j = 0; j < IterMNum; ++j) {
                *(__half2 *)&y[midx + j * MNumPerLoop + nidx * LDM +
                               i * IterNDiff] =
                    TType::compute((__half2 *)x, midx + j * MNumPerLoop,
                                   nidx + i * NNumPerLoop);
            }
        }
    }

    //
    static DEVICE void run(ark::half *c, ark::half *a, ark::half *b, int t0,
                           int t1, int t2)
    {
        // BcastType = 0: both A and B are batched with the same size.
        // BcastType = 1: only A is batched.
        // BcastType = 2: only B is batched.
        static_assert(BcastType == 0 || BcastType == 1 || BcastType == 2,
                      "invalid broadcast type.");
        static_assert(LDM % TDM == 0, "");

        constexpr int LDMN = LDM * LDN;
        static_assert(LDMN % 4 == 0,
                      "detected a potential illegal memory access.");

        constexpr int NNumPerLoop = math::max<1, 2 * TN / TDM>::value;
        constexpr int MNumPerLoop = math::min<TDM, 2 * TN>::value;

        constexpr int IterNDiff = NNumPerLoop * LDM;
        constexpr int IterMNum = math::div_up<TDM, MNumPerLoop>::value;
        constexpr int IterNNum = TDN / NNumPerLoop;

        c = &c[t2 * LDMN];
        if (BcastType == 0) {
            a = &a[t2 * LDMN];
            b = &b[t2 * LDMN];
        } else if (BcastType == 1) {
            a = &a[t2 * LDMN];
        } else if (BcastType == 2) {
            b = &b[t2 * LDMN];
        }

        int midx = BaseOp::midx(t0, 2 * BaseOp::thread_id());
        int nidx = BaseOp::nidx(t1, 2 * BaseOp::thread_id());
#pragma unroll
        for (int i = 0; i < IterNNum; ++i) {
#pragma unroll
            for (int j = 0; j < IterMNum; ++j) {
                *(__half2 *)&c[midx + j * MNumPerLoop + nidx * LDM +
                               i * IterNDiff] =
                    TType::compute((__half2 *)a, (__half2 *)b,
                                   midx + j * MNumPerLoop,
                                   nidx + i * NNumPerLoop);
            }
        }
    }
};

} // namespace ark

#endif // ARK_KERNELS_TRANSFORM_H_
