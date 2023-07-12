// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef ARK_KERNELS_LAYERNORM_H_
#define ARK_KERNELS_LAYERNORM_H_

#include "reduce.h"

namespace ark {

// Static checkers if InShape can be reduced into OutShape.
template <typename InShape, typename OutShape> struct LayerNormShapeChecker
{
    static_assert(InShape::N == OutShape::N,
                  "Dimension N of input and output do not match");
    static_assert(InShape::C == OutShape::C,
                  "Dimension C of input and output do not match");
    static_assert(InShape::H == OutShape::H,
                  "Dimension H of input and output do not match");
    static_assert(OutShape::W == OutShape::W,
                  "Dimension W of input and output do not match");
};

// Perform layer normalization on input and write the result on output.
template <typename InDims, typename InShape, typename OutDims,
          typename OutShape, typename UnitOutShape, int ThreadsNum,
          int SmemBytes, typename DataType, int NelemPerThread>
struct LayerNorm
{
    using UnitOp =
        UnitOp<OutDims, OutShape, UnitOutShape, ThreadsNum, SmemBytes>;

    static_assert(NelemPerThread > 0, "NelemPerThread must be positive");
    static DEVICE void run(DataType *out, DataType *in, int tn, int tc, int th,
                           int tw)
    {
        using InOutChk = LayerNormShapeChecker<InShape, OutShape>;
        using ReduceTypeMean = ReduceTypeMean<DataType, NelemPerThread>;

        constexpr int NonReduceDimLength =
            UnitOutShape::N * UnitOutShape::C * UnitOutShape::H;
        // The reduction dimension of the final stage.
        // Assume this division is always exact.
        static_assert((ThreadsNum * NelemPerThread) % NonReduceDimLength == 0);
        // If we reshape the input into a 2D matrix (NCH x W), ThreadsNum
        // threads compute NCH rows, and each row's sum is computed by
        // ThreadsPerRow threads. If ThreadsPerRow is larger than warp size, we
        // need to use shared memory to reduce the result of each warp.
        constexpr int ThreadsPerRow =
            (ThreadsNum * NelemPerThread) / NonReduceDimLength;

        int tid = UnitOp::thread_id();
        int tid_w = (tid * NelemPerThread) % ThreadsPerRow;
        int tid_h = ((tid * NelemPerThread) / ThreadsPerRow) % UnitOutShape::H;
        int tid_c = ((tid * NelemPerThread) / ThreadsPerRow / UnitOutShape::H) %
                    UnitOutShape::C;
        int tid_n = (tid * NelemPerThread) / ThreadsPerRow / UnitOutShape::H /
                    UnitOutShape::C;

        int idx_in_base =
            (tid_h + th * UnitOutShape::H) * InDims::W +
            (tid_c + tc * UnitOutShape::C) * InDims::W * InDims::H +
            (tid_n + tn * UnitOutShape::N) * InDims::W * InDims::H * InDims::C;

        DataType reduced;
        ReduceTypeMean::singleIdentity(&reduced);
        for (int idx_in_w = tid_w; idx_in_w < InShape::W;
             idx_in_w += ThreadsPerRow) {
            int idx_in = idx_in_base + idx_in_w;
            ReduceTypeMean::singleReduce(&reduced, in[idx_in]);
        }
        ark::sync_warps<ThreadsNum>();
        // final reduction on shared memory using warp shuffle.
        reduced = warpsReduce<ReduceType, UnitOp, ThreadsPerRow>(reduced, tid);
        // get the average result.
        ReduceTypeMean::singlePostReduce(&reduced, &reduced, UnitOutShape::W);
        DataType variance;
        ReduceTypeMean::singleIdentity<DataType>(&variance);
        // get the variance
        ark::sync_warps<ThreadsNum>();
        for (int idx_in_w = tid_w; idx_in_w < InShape::W;
             idx_in_w += ThreadsPerRow) {
            int idx_in = idx_in_base + idx_in_w;
            variance += (in[idx_in] - reduced) * (in[idx_in] - reduced);
        }
        ark::sync_warps<ThreadsNum>();
        variance = warpsReduce<ReduceType, UnitOp, ThreadsPerRow>(variance, tid);
        ReduceTypeMean::singlePostReduce(&variance, &variance, UnitOutShape::W);
        ark::sync_warps<ThreadsNum>();
        // the output is (input - mean) / sqrt(variance)
        for (int idx_in_w = tid_w; idx_in_w < InShape::W;
             idx_in_w += ThreadsPerRow) {
            int idx_in = idx_in_base + idx_in_w;
            out[idx_in] = (in[idx_in] - reduced) * rsqrtf(variance + 1e-5f);
        }
    }
};

template <typename InDims, typename InShape, typename OutDims,
          typename OutShape, typename UnitOutShape, int ThreadsNum,
          int SmemBytes>
DEVICE void layernorm(float *out, float *in, int tx, int ty, int tz)
{
    constexpr int NelemPerThread = 1;
    LayerNorm<InDims, InShape, OutDims, OutShape, UnitOutShape, ThreadsNum,
              SmemBytes, float, NelemPerThread>::run(out, in, tz / OutShape::C,
                                                     tz % OutShape::C, ty, tx);
}

template <typename InDims, typename InShape, typename OutDims,
          typename OutShape, typename UnitOutShape, int ThreadsNum,
          int SmemBytes>
DEVICE void layernorm(ark::half *out, ark::half *in, int tx, int ty, int tz)
{
    constexpr int NelemPerThread = 1;
    LayerNorm<InDims, InShape, OutDims, OutShape, UnitOutShape, ThreadsNum,
              SmemBytes, ark::half, NelemPerThread>::run(out, in,
                                                         tz / OutShape::C,
                                                         tz % OutShape::C, ty,
                                                         tx);
}

} // namespace ark

#endif // ARK_KERNELS_LAYERNORM_H_
