// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef ARK_KERNELS_SOFTMAX_H_
#define ARK_KERNELS_SOFTMAX_H_

#include "reduce.h"

namespace ark {

// Static checkers for Softmax shapes.
template <typename InShape, typename OutShape>
struct SoftmaxShapeChecker {
    static_assert(InShape::N == OutShape::N,
                  "Dimension N of input and output do not match");
    static_assert(InShape::C == OutShape::C,
                  "Dimension C of input and output do not match");
    static_assert(InShape::H == OutShape::H,
                  "Dimension H of input and output do not match");
    static_assert(InShape::W == OutShape::W,
                  "Dimension W of input and output do not match");
};

// Monolithic softmax along the last dimension (W).
// Optimized: single global memory read (register cache), float accumulation,
// fused passes for reduced memory traffic.
// Pass 1: read input → cache in registers, find max
// Pass 2: from registers, compute exp(x - max) and sum (store in registers)
// Pass 3: divide by sum, write output (single global write)
template <typename InDims, typename InShape, typename OutDims,
          typename OutShape, typename UnitOutDims, int NumWarps, int SmemBytes,
          typename DataType, int NelemPerThread>
struct Softmax {
    using UnitOp = UnitOp<OutDims, OutShape, UnitOutDims, NumWarps, SmemBytes>;

    static_assert(NelemPerThread > 0, "NelemPerThread must be positive");

    static DEVICE void run(DataType *out, const DataType *in,
                           int uop_idx, int smem_per_warp) {
        using InOutChk = SoftmaxShapeChecker<InShape, OutShape>;

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

        // --- Pass 1: Read input ONCE, cache in registers, find max ---
        float cached[MaxElemsPerThread];
        float max_val = type::Constant<float>::lowest();
        int num_elems = 0;
#pragma unroll
        for (int idx_w = tid_w; idx_w < InShape::W; idx_w += ThreadsPerRow) {
#pragma unroll
            for (int j = 0; j < NelemPerThread; j++) {
                if (idx_w + j < InShape::W) {
                    float val = type::Cast::compute<float>(in[idx_in_base + idx_w + j]);
                    cached[num_elems] = val;
                    if (val > max_val) max_val = val;
                    num_elems++;
                }
            }
        }

        // Reduce max across warps in float precision
        max_val = warpsReduce<ReduceTypeMax, UnitOp, PhysicalThreadsPerRow>(
            max_val, tid % PhysicalThreadsPerRow, smem_per_warp, warp_offset);

        // --- Pass 2: Compute exp(x - max) from registers, accumulate sum ---
        float sum = 0.0f;
#pragma unroll
        for (int i = 0; i < MaxElemsPerThread; i++) {
            if (i < num_elems) {
                float exp_val = expf(cached[i] - max_val);
                cached[i] = exp_val;  // reuse cache for exp values
                sum += exp_val;
            }
        }

        // Reduce sum across warps in float precision
        sum = warpsReduce<ReduceTypeSum, UnitOp, PhysicalThreadsPerRow>(
            sum, tid % PhysicalThreadsPerRow, smem_per_warp, warp_offset);
        // Note: if all inputs are -inf, sum==0 and output is NaN
        // (matches PyTorch behavior).
        float inv_sum = 1.0f / sum;

        // --- Pass 3: Divide by sum and write output (single global write) ---
        int wi = 0;
#pragma unroll
        for (int idx_w = tid_w; idx_w < InShape::W; idx_w += ThreadsPerRow) {
#pragma unroll
            for (int j = 0; j < NelemPerThread; j++) {
                if (idx_w + j < InShape::W) {
                    out[idx_out_base + idx_w + j] =
                        type::Cast::compute<DataType>(cached[wi] * inv_sum);
                    wi++;
                }
            }
        }
    }
};

// Free function wrapper for softmax.
template <typename InDims, typename InShape, typename OutDims,
          typename OutShape, typename UnitOutDims, int NumWarps, int SmemBytes,
          int NelemPerThread = 1, typename DataType>
DEVICE void softmax(DataType *out, const DataType *in, int uop_idx,
                    int smem_per_warp) {
    Softmax<InDims, InShape, OutDims, OutShape, UnitOutDims, NumWarps,
            SmemBytes, DataType, NelemPerThread>::run(
                out, in, uop_idx, smem_per_warp);
}

}  // namespace ark

#endif  // ARK_KERNELS_SOFTMAX_H_
