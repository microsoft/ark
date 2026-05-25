// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef ARK_KERNELS_MATMUL_H_
#define ARK_KERNELS_MATMUL_H_

#if defined(ARK_TARGET_CUDA_ARCH)
#include "gemm_cutlass.h"
#elif defined(ARK_TARGET_ROCM_ARCH)
#include "gemm_ck.h"
#endif  // defined(ARK_TARGET_ROCM_ARCH)

namespace ark {

/// Matrix multiplication.
///
/// Reuse GEMM kernels. The output is row-major, and the input matrices are
/// row-major by default. If the input matrices are column-major, the
/// corresponding @p IsColumnA or @p IsColumnB should be set to true.
///
/// @tparam OutDims (ark::Vec) Output tensor leading dimensions.
/// @tparam NCA (ark::Vec) A 2D vector with N and C dimensions of matrix A.
/// @tparam NCB (ark::Vec) A 2D vector with N and C dimensions of matrix B.
/// @tparam TileShape (ark::Vec) The output tile shape.
/// @tparam ProblemSize (ark::Vec) The problem size of matmul computation
/// (m, n, k).
/// @tparam LeadingDims (ark::Vec) The leading dimensions of matrix inputs
/// and outputs. (lda, ldc, ldc, ldb).
/// @tparam BatchStrideNA (int)
/// @tparam BatchStrideCA (int)
/// @tparam BatchStrideNB (int)
/// @tparam BatchStrideCB (int)
/// @tparam BatchStrideNC (int)
/// @tparam BatchStrideCC (int)
/// @tparam IsColumnA (bool) Whether matrix A is column-major.
/// @tparam IsColumnB (bool) Whether matrix B is column-major.
/// @tparam NumWarps (int) The number of warps per uop.
/// @tparam SmemBytes (int) The size of shared memory per uop.
///
template <typename OutDims, typename NCA, typename NCB, typename TileShape,
          typename ProblemSize, typename LeadingDims, int BatchStrideNA,
          int BatchStrideCA, int BatchStrideNB, int BatchStrideCB,
          int BatchStrideNC, int BatchStrideCC, bool IsColumnA, bool IsColumnB,
          int NumWarps, int SmemBytes, typename DataTypeA, typename DataTypeB,
          typename DataTypeC>
DEVICE void matmul(DataTypeC *C, DataTypeA *A, DataTypeB *B, int uop_idx,
                   int smem_per_warp) {
    static_assert(NCA::D2 == 1 && NCA::D3 == 1,
                  "NCA should be two dimensional.");
    static_assert(NCB::D2 == 1 && NCB::D3 == 1,
                  "NCB should be two dimensional.");
    static_assert(TileShape::D2 == 1 && TileShape::D3 == 1,
                  "TileShape should be two dimensional.");
    static_assert(ProblemSize::D3 == 1,
                  "ProblemSize should be three dimensional.");

    // N dimension of C is max(N dimension of A, N dimension of B)
    constexpr int NC = (NCA::D0 > NCB::D0) ? NCA::D0 : NCB::D0;
    // C dimension of C is max(C dimension of A, C dimension of B)
    constexpr int CC = (NCA::D1 > NCB::D1) ? NCA::D1 : NCB::D1;

    using OutShape = Vec<NC, CC, ProblemSize::D0, ProblemSize::D1>;
    using UnitOutDims = Vec<1, 1, TileShape::D0, TileShape::D1>;
    using UnitOp = UnitOp<OutDims, OutShape, UnitOutDims, NumWarps, SmemBytes>;

    constexpr int LeadingDimA = LeadingDims::D0;
    constexpr int LeadingDimB = LeadingDims::D3;
    constexpr int LeadingDimC = LeadingDims::D1;
    constexpr int ProblemSizeM = ProblemSize::D0;
    constexpr int ProblemSizeN = ProblemSize::D1;
    constexpr int ProblemSizeK = ProblemSize::D2;
    constexpr int TileSizeM = TileShape::D0;
    constexpr int TileSizeN = TileShape::D1;

    int un = UnitOp::uop_idx_n(uop_idx);
    int uc = UnitOp::uop_idx_c(uop_idx);

    // Broadcasting
    DataTypeA *pA = &A[un * BatchStrideNA + uc * BatchStrideCA];
    DataTypeB *pB = &B[un * BatchStrideNB + uc * BatchStrideCB];
    DataTypeC *pC = &C[un * BatchStrideNC + uc * BatchStrideCC];

#if defined(ARK_TARGET_CUDA_ARCH)
    gemm_cutlass<DataTypeA, LeadingDimA, IsColumnA, DataTypeB, LeadingDimB,
                 IsColumnB, DataTypeC, LeadingDimC, ProblemSizeM, ProblemSizeN,
                 ProblemSizeK, TileSizeM, TileSizeN, UnitOp>(
        pC, pA, pB, uop_idx, smem_per_warp);
#elif defined(ARK_TARGET_ROCM_ARCH)
    gemm_ck<DataTypeA, LeadingDimA, IsColumnA, DataTypeB, LeadingDimB,
            IsColumnB, DataTypeC, LeadingDimC, ProblemSizeM, ProblemSizeN,
            ProblemSizeK, TileSizeM, TileSizeN, UnitOp>(pC, pA, pB, uop_idx,
                                                        smem_per_warp);
#endif
}

/// Matrix multiplication with GELU activation fused into the epilogue.
/// Output = gelu(A @ B). Only supported on CUDA (CUTLASS).
template <typename OutDims, typename NCA, typename NCB, typename TileShape,
          typename ProblemSize, typename LeadingDims, int BatchStrideNA,
          int BatchStrideCA, int BatchStrideNB, int BatchStrideCB,
          int BatchStrideNC, int BatchStrideCC, bool IsColumnA, bool IsColumnB,
          int NumWarps, int SmemBytes, typename DataTypeA, typename DataTypeB,
          typename DataTypeC>
DEVICE void matmul_gelu(DataTypeC *C, DataTypeA *A, DataTypeB *B, int uop_idx,
                        int smem_per_warp) {
    static_assert(NCA::D2 == 1 && NCA::D3 == 1,
                  "NCA should be two dimensional.");
    static_assert(NCB::D2 == 1 && NCB::D3 == 1,
                  "NCB should be two dimensional.");
    static_assert(TileShape::D2 == 1 && TileShape::D3 == 1,
                  "TileShape should be two dimensional.");
    static_assert(ProblemSize::D3 == 1,
                  "ProblemSize should be three dimensional.");

    constexpr int NC = (NCA::D0 > NCB::D0) ? NCA::D0 : NCB::D0;
    constexpr int CC = (NCA::D1 > NCB::D1) ? NCA::D1 : NCB::D1;

    using OutShape = Vec<NC, CC, ProblemSize::D0, ProblemSize::D1>;
    using UnitOutDims = Vec<1, 1, TileShape::D0, TileShape::D1>;
    using UnitOp = UnitOp<OutDims, OutShape, UnitOutDims, NumWarps, SmemBytes>;

    constexpr int LeadingDimA = LeadingDims::D0;
    constexpr int LeadingDimB = LeadingDims::D3;
    constexpr int LeadingDimC = LeadingDims::D1;
    constexpr int ProblemSizeM = ProblemSize::D0;
    constexpr int ProblemSizeN = ProblemSize::D1;
    constexpr int ProblemSizeK = ProblemSize::D2;
    constexpr int TileSizeM = TileShape::D0;
    constexpr int TileSizeN = TileShape::D1;

    int un = UnitOp::uop_idx_n(uop_idx);
    int uc = UnitOp::uop_idx_c(uop_idx);

    DataTypeA *pA = &A[un * BatchStrideNA + uc * BatchStrideCA];
    DataTypeB *pB = &B[un * BatchStrideNB + uc * BatchStrideCB];
    DataTypeC *pC = &C[un * BatchStrideNC + uc * BatchStrideCC];

#if defined(ARK_TARGET_CUDA_ARCH)
    gemm_cutlass_gelu<DataTypeA, LeadingDimA, IsColumnA, DataTypeB,
                      LeadingDimB, IsColumnB, DataTypeC, LeadingDimC,
                      ProblemSizeM, ProblemSizeN, ProblemSizeK, TileSizeM,
                      TileSizeN, UnitOp>(pC, pA, pB, uop_idx, smem_per_warp);
#elif defined(ARK_TARGET_ROCM_ARCH)
    static_assert(false, "matmul_gelu not supported on ROCm.");
#endif
}

/// Matrix multiplication with residual addition fused into the epilogue.
/// Output = A @ B + Residual. Only supported on CUDA (CUTLASS).
template <typename OutDims, typename NCA, typename NCB, typename TileShape,
          typename ProblemSize, typename LeadingDims, int BatchStrideNA,
          int BatchStrideCA, int BatchStrideNB, int BatchStrideCB,
          int BatchStrideNC, int BatchStrideCC, bool IsColumnA, bool IsColumnB,
          int NumWarps, int SmemBytes, typename DataTypeA, typename DataTypeB,
          typename DataTypeC>
DEVICE void matmul_add(DataTypeC *D, DataTypeA *A, DataTypeB *B,
                       DataTypeC *Residual, int uop_idx, int smem_per_warp) {
    static_assert(NCA::D2 == 1 && NCA::D3 == 1,
                  "NCA should be two dimensional.");
    static_assert(NCB::D2 == 1 && NCB::D3 == 1,
                  "NCB should be two dimensional.");
    static_assert(TileShape::D2 == 1 && TileShape::D3 == 1,
                  "TileShape should be two dimensional.");
    static_assert(ProblemSize::D3 == 1,
                  "ProblemSize should be three dimensional.");

    constexpr int NC = (NCA::D0 > NCB::D0) ? NCA::D0 : NCB::D0;
    constexpr int CC = (NCA::D1 > NCB::D1) ? NCA::D1 : NCB::D1;

    using OutShape = Vec<NC, CC, ProblemSize::D0, ProblemSize::D1>;
    using UnitOutDims = Vec<1, 1, TileShape::D0, TileShape::D1>;
    using UnitOp = UnitOp<OutDims, OutShape, UnitOutDims, NumWarps, SmemBytes>;

    constexpr int LeadingDimA = LeadingDims::D0;
    constexpr int LeadingDimB = LeadingDims::D3;
    constexpr int LeadingDimC = LeadingDims::D1;
    constexpr int ProblemSizeM = ProblemSize::D0;
    constexpr int ProblemSizeN = ProblemSize::D1;
    constexpr int ProblemSizeK = ProblemSize::D2;
    constexpr int TileSizeM = TileShape::D0;
    constexpr int TileSizeN = TileShape::D1;

    int un = UnitOp::uop_idx_n(uop_idx);
    int uc = UnitOp::uop_idx_c(uop_idx);

    DataTypeA *pA = &A[un * BatchStrideNA + uc * BatchStrideCA];
    DataTypeB *pB = &B[un * BatchStrideNB + uc * BatchStrideCB];
    DataTypeC *pD = &D[un * BatchStrideNC + uc * BatchStrideCC];
    DataTypeC *pR = &Residual[un * BatchStrideNC + uc * BatchStrideCC];

#if defined(ARK_TARGET_CUDA_ARCH)
    gemm_cutlass_add<DataTypeA, LeadingDimA, IsColumnA, DataTypeB, LeadingDimB,
                     IsColumnB, DataTypeC, LeadingDimC, ProblemSizeM,
                     ProblemSizeN, ProblemSizeK, TileSizeM, TileSizeN, UnitOp>(
        pD, pA, pB, pR, uop_idx, smem_per_warp);
#elif defined(ARK_TARGET_ROCM_ARCH)
    static_assert(false, "matmul_add not supported on ROCm.");
#endif
}

}  // namespace ark

// Fused matmul with register-level functor (scale, gelu, relu, etc.)
#if defined(ARK_TARGET_CUDA_ARCH)
#include "gemm_fused.h"

namespace ark {

/// Matrix multiplication with scale: D = scale * (A @ B).
/// The scale is applied on register accumulators via FunctorScale BEFORE
/// the epilogue writes to global memory — zero extra global memory traffic.
template <typename OutDims, typename NCA, typename NCB, typename TileShape,
          typename ProblemSize, typename LeadingDims, int BatchStrideNA,
          int BatchStrideCA, int BatchStrideNB, int BatchStrideCB,
          int BatchStrideNC, int BatchStrideCC, bool IsColumnA, bool IsColumnB,
          int NumWarps, int SmemBytes, uint32_t ScaleBits,
          typename DataTypeA, typename DataTypeB, typename DataTypeC>
DEVICE void matmul_scale(DataTypeC *C, DataTypeA *A, DataTypeB *B,
                         int uop_idx, int smem_per_warp) {
    static_assert(NCA::D2 == 1 && NCA::D3 == 1);
    static_assert(NCB::D2 == 1 && NCB::D3 == 1);
    static_assert(TileShape::D2 == 1 && TileShape::D3 == 1);
    static_assert(ProblemSize::D3 == 1);

    constexpr int NC = (NCA::D0 > NCB::D0) ? NCA::D0 : NCB::D0;
    constexpr int CC = (NCA::D1 > NCB::D1) ? NCA::D1 : NCB::D1;
    using OutShape = Vec<NC, CC, ProblemSize::D0, ProblemSize::D1>;
    using UnitOutDims = Vec<1, 1, TileShape::D0, TileShape::D1>;
    using UnitOp_t = UnitOp<OutDims, OutShape, UnitOutDims, NumWarps, SmemBytes>;

    int un = UnitOp_t::uop_idx_n(uop_idx);
    int uc = UnitOp_t::uop_idx_c(uop_idx);
    DataTypeA *pA = &A[un * BatchStrideNA + uc * BatchStrideCA];
    DataTypeB *pB = &B[un * BatchStrideNB + uc * BatchStrideCB];
    DataTypeC *pC = &C[un * BatchStrideNC + uc * BatchStrideCC];

    // Decode scale from bit-pattern template parameter
    union { uint32_t u; float f; } conv;
    conv.u = ScaleBits;
    FunctorScale functor{conv.f};

    gemm_cutlass_fused<
        typename std::remove_const<DataTypeA>::type, LeadingDims::D0, IsColumnA,
        typename std::remove_const<DataTypeB>::type, LeadingDims::D3, IsColumnB,
        DataTypeC, LeadingDims::D1,
        ProblemSize::D0, ProblemSize::D1, ProblemSize::D2,
        TileShape::D0, TileShape::D1,
        UnitOp_t, FunctorScale>(pC, pA, pB, functor, uop_idx, smem_per_warp);
}

}  // namespace ark
#endif  // ARK_TARGET_CUDA_ARCH

#endif  // ARK_KERNELS_MATMUL_H_
