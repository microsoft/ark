// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef ARK_KERNELS_GEMV_H_
#define ARK_KERNELS_GEMV_H_

#include "common/unit_op.h"

namespace ark {

/// M==1 GEMV: C[0, n] = sum_k A[0, k] * B[n, k] for n in this uop's column tile.
///
/// Fast path for matmul with a [1, K] input (decode / GEMV shapes). The CUTLASS
/// GEMM pads M=1 up to a 64-row tile and runs the full MMA (64x wasted compute);
/// this kernel computes exactly one output row, so it is memory-bound on B (the
/// weight, read once) — the optimal GEMV cost.
///
/// Linear-layer layout only (routed when !trans_input && trans_other): A is
/// [1, K] row-major (leading dim K), B ("other") is [N, K] row-major (leading
/// dim K), C is [1, N] row-major. One warp per output column: the 32 lanes
/// split the K reduction with coalesced int4 loads, then warp-reduce in fp32.
///
/// Template params mirror the matmul kernel's forwarding (see matmul.h).
template <typename DataTypeA, int LeadingDimA, bool IsColumnA,
          typename DataTypeB, int LeadingDimB, bool IsColumnB,
          typename DataTypeC, int LeadingDimC, int ProblemSizeM,
          int ProblemSizeN, int ProblemSizeK, int TileSizeM, int TileSizeN,
          typename UnitOp>
DEVICE void gemv(DataTypeC *C, DataTypeA *A, DataTypeB *B, int uop_idx,
                 int /*smem_per_warp*/) {
    static_assert(TileSizeM == 1, "gemv: output tile must have M==1");
    static_assert(sizeof(DataTypeA) == 2 && sizeof(DataTypeB) == 2,
                  "gemv supports 2-byte A/B (fp16/bf16)");
    constexpr int VEC = sizeof(int4) / sizeof(DataTypeA);  // 8 for bf16/fp16
    static_assert(ProblemSizeK % VEC == 0,
                  "gemv: K must be a multiple of the int4 width (8)");
    constexpr int Kv = ProblemSizeK / VEC;

    const int n0 = UnitOp::uop_idx_w(uop_idx) * TileSizeN;
    const int tid = UnitOp::thread_id();
    const int nThreads = UnitOp::NumThreads;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int nWarps = nThreads >> 5;

    const int4 *Av = reinterpret_cast<const int4 *>(A);  // A[0, 0..K)
    // Each warp owns output column(s); lanes split the K reduction.
    for (int col = warp; col < TileSizeN; col += nWarps) {
        const int n = n0 + col;
        if (n >= ProblemSizeN) continue;
        const int4 *Bv = reinterpret_cast<const int4 *>(
            B + static_cast<size_t>(n) * LeadingDimB);
        float acc = 0.0f;
        for (int kv = lane; kv < Kv; kv += 32) {
            int4 a4 = Av[kv];
            int4 b4 = Bv[kv];
            DataTypeA *ap = reinterpret_cast<DataTypeA *>(&a4);
            DataTypeB *bp = reinterpret_cast<DataTypeB *>(&b4);
#pragma unroll
            for (int e = 0; e < VEC; ++e) {
                acc += static_cast<float>(ap[e]) * static_cast<float>(bp[e]);
            }
        }
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) {
            acc += __shfl_down_sync(0xffffffffu, acc, o);
        }
        if (lane == 0) {
            C[n] = static_cast<DataTypeC>(acc);  // C[0, n] (row 0)
        }
    }
}

}  // namespace ark

#endif  // ARK_KERNELS_GEMV_H_
