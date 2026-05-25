// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.
//
// matmul_fused: matmul with a post-MMA functor applied on register accumulators.
// Wraps gemm_fused.h for ARK's op interface.

#ifndef ARK_KERNELS_MATMUL_FUSED_H_
#define ARK_KERNELS_MATMUL_FUSED_H_

#include "gemm_fused.h"

namespace ark {

/// Matrix multiplication with scale applied on register accumulators.
/// Output = (A @ B) * scale
/// The scale is applied BEFORE writing to global memory, saving one
/// global read+write cycle compared to separate matmul + scale ops.
template <typename OutDims, typename NCA, typename NCB, typename TileShape,
          typename ProblemSize, typename LeadingDims, int BatchStrideNA,
          int BatchStrideCA, int BatchStrideNB, int BatchStrideCB,
          int BatchStrideNC, int BatchStrideCC, bool IsColumnA, bool IsColumnB,
          int NumWarps, int SmemBytes, typename DataTypeA, typename DataTypeB,
          typename DataTypeC>
DEVICE void matmul_scale(DataTypeC *C, DataTypeA *A, DataTypeB *B,
                         float scale, int uop_idx, int smem_per_warp) {
    constexpr int NC = (NCA::D0 > NCB::D0) ? NCA::D0 : NCB::D0;
    constexpr int CC = (NCA::D1 > NCB::D1) ? NCA::D1 : NCB::D1;
    using OutShape = Vec<NC, CC, ProblemSize::D0, ProblemSize::D1>;
    using UnitOutDims = Vec<1, 1, TileShape::D0, TileShape::D1>;
    using UnitOp_t = UnitOp<OutDims, OutShape, UnitOutDims, NumWarps, SmemBytes>;

    constexpr int LeadingDimA = LeadingDims::D0;
    constexpr int LeadingDimB = LeadingDims::D3;
    constexpr int LeadingDimC = LeadingDims::D1;

    int un = UnitOp_t::uop_idx_n(uop_idx);
    int uc = UnitOp_t::uop_idx_c(uop_idx);

    DataTypeA *pA = &A[un * BatchStrideNA + uc * BatchStrideCA];
    DataTypeB *pB = &B[un * BatchStrideNB + uc * BatchStrideCB];
    DataTypeC *pC = &C[un * BatchStrideNC + uc * BatchStrideCC];

    FunctorScale functor{scale};
    gemm_with_functor<
        typename std::remove_const<DataTypeA>::type, LeadingDimA, IsColumnA,
        typename std::remove_const<DataTypeB>::type, LeadingDimB, IsColumnB,
        DataTypeC, LeadingDimC,
        ProblemSize::D0, ProblemSize::D1, ProblemSize::D2,
        TileShape::D0, TileShape::D1,
        UnitOp_t, FunctorScale>(pC, pA, pB, functor, uop_idx, smem_per_warp);
}

}  // namespace ark

#endif  // ARK_KERNELS_MATMUL_FUSED_H_
