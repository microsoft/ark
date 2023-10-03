// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef ARK_KERNELS_LAYERNORM_H_
#define ARK_KERNELS_LAYERNORM_H_

#include "reduce.h"

namespace ark {

// Static checkers if InShape can be reduced into OutShape.
template <typename InShape, typename OutShape>
struct LayerNormShapeChecker {
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
struct LayerNorm {
    using UnitOp =
        UnitOp<OutDims, OutShape, UnitOutDims, NumThreads, SmemBytes>;

    static_assert(NelemPerThread > 0, "NelemPerThread must be positive");
    static DEVICE void run(DataType *out, const DataType *in, int uop_idx,
                           int smem_per_warp) {
        using InOutChk = LayerNormShapeChecker<InShape, OutShape>;

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

        int un = UnitOp::uop_idx_n(uop_idx);
        int uc = UnitOp::uop_idx_c(uop_idx);
        int uh = UnitOp::uop_idx_h(uop_idx);

        int idx_out_base = (tid_h + uh * UnitOutDims::H) * OutDims::W +
                           (tid_c + uc * UnitOutDims::C) * OutDims::HW +
                           (tid_n + un * UnitOutDims::N) * OutDims::CHW;
        int idx_in_base = (tid_h + uh * UnitOutDims::H) * InDims::W +
                          (tid_c + uc * UnitOutDims::C) * InDims::HW +
                          (tid_n + un * UnitOutDims::N) * InDims::CHW;

        DataType reduced;
        ReduceTypeMean::identity<1>(&reduced);
        for (int idx_in_w = tid_w; idx_in_w < InShape::W;
             idx_in_w += ThreadsPerRow) {
            int idx_in = idx_in_base + idx_in_w;
            ReduceTypeMean::reduce<1>(&reduced, &reduced, &in[idx_in]);
        }
        // final reduction on shared memory using warp shuffle.
        reduced = warpsReduce<ReduceTypeMean, UnitOp, ThreadsPerRow>(
            reduced, tid, smem_per_warp);
        // get the average result.
        ReduceTypeMean::postReduce<1>(&reduced, &reduced, UnitOutDims::W);
        DataType variance;
        ReduceTypeMean::identity<1>(&variance);
        // get the variance
        // TODO: Kahan sum
        for (int idx_in_w = tid_w; idx_in_w < InShape::W;
             idx_in_w += ThreadsPerRow) {
            int idx_in = idx_in_base + idx_in_w;
            variance += (in[idx_in] - reduced) * (in[idx_in] - reduced);
        }
        variance = warpsReduce<ReduceTypeMean, UnitOp, ThreadsPerRow>(
            variance, tid, smem_per_warp);
        ReduceTypeMean::postReduce<1>(&variance, &variance, UnitOutDims::W);
        // the output is (input - mean) / sqrt(variance)
        for (int idx_w = tid_w; idx_w < InShape::W; idx_w += ThreadsPerRow) {
            int idx_in = idx_in_base + idx_w;
            int idx_out = idx_out_base + idx_w;
            out[idx_out] = (in[idx_in] - reduced) * rsqrtf(variance + 1e-5f);
        }
    }
};

template <typename InDims, typename InShape, typename OutDims,
          typename OutShape, typename UnitOutDims, int NumThreads,
          int SmemBytes, typename DataType>
DEVICE void layernorm(DataType *out, const DataType *in, int uop_idx,
                      int smem_per_warp) {
    constexpr int NelemPerThread = 1;
    LayerNorm<InDims, InShape, OutDims, OutShape, UnitOutDims, NumThreads,
              SmemBytes, DataType, NelemPerThread>::run(out, in, uop_idx,
                                                        smem_per_warp);
}

// Perform RMS normalization on input and write the result on output.
// Root Mean Square Layer Normalization: https://arxiv.org/pdf/1910.07467.pdf
template <typename InDims, typename InShape, typename OutDims,
          typename OutShape, typename UnitOutDims, int NumThreads,
          int SmemBytes, typename DataType, int NelemPerThread>
struct RMSNorm {
    using UnitOp =
        UnitOp<OutDims, OutShape, UnitOutDims, NumThreads, SmemBytes>;

    static_assert(NelemPerThread > 0, "NelemPerThread must be positive");
    static DEVICE void run(DataType *out, const DataType *in, int uop_idx,
                           int smem_per_warp) {
        using InOutChk = LayerNormShapeChecker<InShape, OutShape>;

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

        int un = UnitOp::uop_idx_n(uop_idx);
        int uc = UnitOp::uop_idx_c(uop_idx);
        int uh = UnitOp::uop_idx_h(uop_idx);

        int idx_out_base = (tid_h + uh * UnitOutDims::H) * OutDims::W +
                           (tid_c + uc * UnitOutDims::C) * OutDims::HW +
                           (tid_n + un * UnitOutDims::N) * OutDims::CHW;
        int idx_in_base = (tid_h + uh * UnitOutDims::H) * InDims::W +
                          (tid_c + uc * UnitOutDims::C) * InDims::HW +
                          (tid_n + un * UnitOutDims::N) * InDims::CHW;

        // calculate mean square
        DataType mean_square;
        DataType cmp;
        ReduceTypeMean::identity<1>(&mean_square);
        ReduceTypeMean::identity<1>(&cmp);
        for (int idx_in_w = tid_w; idx_in_w < InShape::W;
             idx_in_w += ThreadsPerRow) {
            int idx_in = idx_in_base + idx_in_w;
            DataType in_val = in[idx_in];
            DataType val = in_val * in_val - cmp;
            DataType tmp = mean_square + val;
            cmp = (tmp - mean_square) - val;
            mean_square = tmp;
        }
        mean_square = warpsReduce<ReduceTypeMean, UnitOp, ThreadsPerRow>(
            mean_square, tid, smem_per_warp);
        ReduceTypeMean::postReduce<1>(&mean_square, &mean_square,
                                      UnitOutDims::W);
        // the output is (input - mean) / sqrt(mean_square)
        DataType rrms(rsqrtf(mean_square + 1e-5f));
        for (int idx_w = tid_w; idx_w < InShape::W; idx_w += ThreadsPerRow) {
            int idx_in = idx_in_base + idx_w;
            int idx_out = idx_out_base + idx_w;
            out[idx_out] = (in[idx_in]) * rrms;
        }
    }
};

template <typename InDims, typename InShape, typename OutDims,
          typename OutShape, typename UnitOutDims, int NumThreads,
          int SmemBytes, typename DataType>
DEVICE void rmsnorm(DataType *out, const DataType *in, int uop_idx,
                    int smem_per_warp) {
    constexpr int NelemPerThread = 1;
    RMSNorm<InDims, InShape, OutDims, OutShape, UnitOutDims, NumThreads,
            SmemBytes, DataType, NelemPerThread>::run(out, in, uop_idx,
                                                      smem_per_warp);
}

}  // namespace ark

#endif  // ARK_KERNELS_LAYERNORM_H_
