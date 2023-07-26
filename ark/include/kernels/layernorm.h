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
          typename OutShape, typename UnitOutDims, int NumThreads,
          int SmemBytes, typename DataType, int NelemPerThread>
struct LayerNorm
{
    using UnitOp =
        UnitOp<OutDims, OutShape, UnitOutDims, NumThreads, SmemBytes>;

    static_assert(NelemPerThread > 0, "NelemPerThread must be positive");
    static DEVICE void run(DataType *out, const DataType *in, int tn, int tc,
                           int th, int tw)
    {
        using InOutChk = LayerNormShapeChecker<InShape, OutShape>;
        using ReduceTypeMean = ReduceTypeMean<DataType, NelemPerThread>;

        constexpr int NonReduceDimLength = UnitOutDims::NCH;
        // The reduction dimension of the final stage.
        // Assume this division is always exact.
        static_assert((NumThreads * NelemPerThread) % NonReduceDimLength == 0);
        // If we reshape the input into a 2D matrix (NCH x W), NumThreads
        // threads compute NCH rows, and each row's sum is computed by
        // ThreadsPerRow threads. If ThreadsPerRow is larger than warp size, we
        // need to use shared memory to reduce the result of each warp.
        constexpr int ThreadsPerRow =
            (NumThreads * NelemPerThread) / NonReduceDimLength;

        int tid = UnitOp::thread_id();
        int tid_w = (tid * NelemPerThread) % ThreadsPerRow;
        int tid_h = ((tid * NelemPerThread) / ThreadsPerRow) % UnitOutDims::H;
        int tid_c = ((tid * NelemPerThread) / ThreadsPerRow / UnitOutDims::H) %
                    UnitOutDims::C;
        int tid_n = (tid * NelemPerThread) / ThreadsPerRow / UnitOutDims::CH;

        int idx_in_base = (tid_h + th * UnitOutDims::H) * InDims::W +
                          (tid_c + tc * UnitOutDims::C) * InDims::HW +
                          (tid_n + tn * UnitOutDims::N) * InDims::CHW;

        DataType reduced;
        ReduceTypeMean::singleIdentity(&reduced);
        for (int idx_in_w = tid_w; idx_in_w < InShape::W;
             idx_in_w += ThreadsPerRow) {
            int idx_in = idx_in_base + idx_in_w;
            ReduceTypeMean::singleReduce(&reduced, &reduced, &in[idx_in]);
        }
        ark::sync_warps<NumThreads>();
        // final reduction on shared memory using warp shuffle.
        reduced =
            warpsReduce<ReduceTypeMean, UnitOp, ThreadsPerRow>(reduced, tid);
        // get the average result.
        ReduceTypeMean::singlePostReduce(&reduced, &reduced, UnitOutDims::W);
        DataType variance;
        ReduceTypeMean::singleIdentity(&variance);
        // get the variance
        ark::sync_warps<NumThreads>();
        for (int idx_in_w = tid_w; idx_in_w < InShape::W;
             idx_in_w += ThreadsPerRow) {
            int idx_in = idx_in_base + idx_in_w;
            variance += (in[idx_in] - reduced) * (in[idx_in] - reduced);
        }
        ark::sync_warps<NumThreads>();
        variance =
            warpsReduce<ReduceTypeMean, UnitOp, ThreadsPerRow>(variance, tid);
        ReduceTypeMean::singlePostReduce(&variance, &variance, UnitOutDims::W);
        ark::sync_warps<NumThreads>();
        // the output is (input - mean) / sqrt(variance)
        for (int idx_in_w = tid_w; idx_in_w < InShape::W;
             idx_in_w += ThreadsPerRow) {
            int idx_in = idx_in_base + idx_in_w;
            out[idx_in] = (in[idx_in] - reduced) * rsqrtf(variance + 1e-5f);
        }
    }
};

template <typename InDims, typename InShape, typename OutDims,
          typename OutShape, typename UnitOutDims, int NumThreads,
          int SmemBytes>
DEVICE void layernorm(float *out, const float *in, int tx, int ty, int tz)
{
    constexpr int NelemPerThread = 1;
    LayerNorm<InDims, InShape, OutDims, OutShape, UnitOutDims, NumThreads,
              SmemBytes, float, NelemPerThread>::run(out, in, tz / OutShape::C,
                                                     tz % OutShape::C, ty, tx);
}

template <typename InDims, typename InShape, typename OutDims,
          typename OutShape, typename UnitOutDims, int NumThreads,
          int SmemBytes>
DEVICE void layernorm(ark::half *out, const ark::half *in, int tx, int ty,
                      int tz)
{
    constexpr int NelemPerThread = 1;
    LayerNorm<InDims, InShape, OutDims, OutShape, UnitOutDims, NumThreads,
              SmemBytes, ark::half, NelemPerThread>::run(out, in,
                                                         tz / OutShape::C,
                                                         tz % OutShape::C, ty,
                                                         tx);
}

} // namespace ark

#endif // ARK_KERNELS_LAYERNORM_H_
