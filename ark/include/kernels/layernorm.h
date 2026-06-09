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
    static_assert(InShape::W == OutShape::W,
                  "Dimension W of input and output do not match");
};

// Perform layer normalization on input and write the result on output.
// When HasGammaBeta is true, applies affine transform: gamma * normalized + beta.
// gamma and beta are 1-D tensors of size W (the normalization dimension).
//
// Optimized: single global memory read (register cache), float accumulation,
// multi-element-per-thread unrolling for reduced loop overhead.
template <typename InDims, typename InShape, typename OutDims,
          typename OutShape, typename UnitOutDims, int NumWarps, int SmemBytes,
          typename DataType, int NelemPerThread, bool HasGammaBeta>
struct LayerNorm {
    using UnitOp = UnitOp<OutDims, OutShape, UnitOutDims, NumWarps, SmemBytes>;

    static_assert(NelemPerThread > 0, "NelemPerThread must be positive");
    static DEVICE void run(DataType *out, const DataType *in,
                           const DataType *gamma, const DataType *beta,
                           int uop_idx, int smem_per_warp) {
        using InOutChk = LayerNormShapeChecker<InShape, OutShape>;
        static_assert(sizeof(InOutChk) > 0, "");

        constexpr int NonReduceDimLength = UnitOutDims::NCH;
        static_assert(
            (UnitOp::NumThreads * NelemPerThread) % NonReduceDimLength == 0);
        static_assert(UnitOp::NumThreads % NonReduceDimLength == 0,
                      "NumThreads must be evenly divisible by "
                      "NonReduceDimLength for correct physical "
                      "thread-to-row assignment");
        constexpr int ThreadsPerRow =
            (UnitOp::NumThreads * NelemPerThread) / NonReduceDimLength;

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

        UnitOp::sync_threads();

        // Compute warp_offset for multi-row shared memory partitioning
        constexpr int PhysicalThreadsPerRow =
            UnitOp::NumThreads / NonReduceDimLength;
        static_assert(PhysicalThreadsPerRow > 0,
                      "Not enough threads for tile dimensions");
        static_assert(PhysicalThreadsPerRow <= Arch::ThreadsPerWarp ||
                          PhysicalThreadsPerRow % Arch::ThreadsPerWarp == 0,
                      "PhysicalThreadsPerRow must be <= warp size or a "
                      "multiple of warp size");
        constexpr int WarpsPerRow = PhysicalThreadsPerRow / Arch::ThreadsPerWarp;
        static_assert(WarpsPerRow * NonReduceDimLength <= Arch::ThreadsPerWarp,
                      "Too many warps for ReduceSharedStorage capacity");
        int row_in_tile = tid / PhysicalThreadsPerRow;
        int warp_offset = row_in_tile * WarpsPerRow;

        constexpr int OuterIters =
            (InShape::W + ThreadsPerRow - 1) / ThreadsPerRow;
        constexpr int MaxElemsPerThread = OuterIters * NelemPerThread;

        // --- Pass 1: Read input ONCE from global memory, cache in registers,
        //             accumulate sum for mean (float accumulation) ---
        float cached[MaxElemsPerThread];
        float sum = 0.0f;
        int num_elems = 0;
#pragma unroll
        for (int idx_w = tid_w; idx_w < InShape::W; idx_w += ThreadsPerRow) {
#pragma unroll
            for (int j = 0; j < NelemPerThread; j++) {
                if (idx_w + j < InShape::W) {
                    float val = type::Cast::compute<float>(in[idx_in_base + idx_w + j]);
                    cached[num_elems] = val;
                    sum += val;
                    num_elems++;
                }
            }
        }

        // Reduce sum across physical threads (each thread already accumulated
        // NelemPerThread elements locally, so we reduce PhysicalThreadsPerRow threads).
        sum = warpsReduce<ReduceTypeSum, UnitOp, PhysicalThreadsPerRow>(
            sum, tid % PhysicalThreadsPerRow, smem_per_warp, warp_offset);
        float fmean = sum / static_cast<float>(InShape::W);

        // --- Pass 2: Compute variance from cached registers (no global read) ---
        float var_sum = 0.0f;
#pragma unroll
        for (int i = 0; i < MaxElemsPerThread; i++) {
            if (i < num_elems) {
                float diff = cached[i] - fmean;
                var_sum += diff * diff;
            }
        }

        var_sum = warpsReduce<ReduceTypeSum, UnitOp, PhysicalThreadsPerRow>(
            var_sum, tid % PhysicalThreadsPerRow, smem_per_warp, warp_offset);
        float inv_std = rsqrtf(var_sum / static_cast<float>(InShape::W) + 1e-5f);

        // --- Pass 3: Normalize and write output (from registers) ---
        int wi = 0;
#pragma unroll
        for (int idx_w = tid_w; idx_w < InShape::W; idx_w += ThreadsPerRow) {
#pragma unroll
            for (int j = 0; j < NelemPerThread; j++) {
                if (idx_w + j < InShape::W) {
                    float normalized = (cached[wi] - fmean) * inv_std;
                    if constexpr (HasGammaBeta) {
                        normalized = normalized *
                            type::Cast::compute<float>(gamma[idx_w + j]) +
                            type::Cast::compute<float>(beta[idx_w + j]);
                    }
                    out[idx_out_base + idx_w + j] = type::Cast::compute<DataType>(normalized);
                    wi++;
                }
            }
        }
    }
};

// Free function for layernorm without gamma/beta. Currently unused by the op
// layer (which always uses layernorm_affine), but retained for kernel-level API
// completeness and potential future use (e.g., non-affine LayerNorm op).
template <typename InDims, typename InShape, typename OutDims,
          typename OutShape, typename UnitOutDims, int NumWarps, int SmemBytes,
          int NelemPerThread = 1, typename DataType>
DEVICE void layernorm(DataType *out, const DataType *in, int uop_idx,
                      int smem_per_warp) {
    LayerNorm<InDims, InShape, OutDims, OutShape, UnitOutDims, NumWarps,
              SmemBytes, DataType, NelemPerThread, false>::run(
                  out, in, nullptr, nullptr, uop_idx, smem_per_warp);
}

// Free function for layernorm with gamma/beta affine transform.
template <typename InDims, typename InShape, typename OutDims,
          typename OutShape, typename UnitOutDims, int NumWarps, int SmemBytes,
          int NelemPerThread = 1, typename DataType>
DEVICE void layernorm_affine(DataType *out, const DataType *in,
                             const DataType *gamma, const DataType *beta,
                             int uop_idx, int smem_per_warp) {
    LayerNorm<InDims, InShape, OutDims, OutShape, UnitOutDims, NumWarps,
              SmemBytes, DataType, NelemPerThread, true>::run(
                  out, in, gamma, beta, uop_idx, smem_per_warp);
}

}  // namespace ark

#endif  // ARK_KERNELS_LAYERNORM_H_
