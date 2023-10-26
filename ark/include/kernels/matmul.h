// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef ARK_KERNELS_MATMUL_H_
#define ARK_KERNELS_MATMUL_H_

#if defined(ARK_TARGET_CUDA_ARCH)
#include "gemm_cutlass.h"
#elif defined(ARK_TARGET_ROCM_ARCH)
#include "gemm_ck.h"
#endif
#include "unit_op.h"

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
/// @tparam Shape (ark::Vec) The tile shape of matmul computation (m, n, k).
/// @tparam ProblemSize (ark::Vec) The problem size of matmul computation
/// (m, n, k).
/// @tparam LeadingDims (ark::Vec) The leading dimensions of matrix inputs
/// and outputs. (lda, ldc, ldc, ldb).
/// @tparam InnerLdimA (int) The leading dimension of the inner dimension of A.
/// @tparam InnerLdimB (int) The leading dimension of the inner dimension of B.
/// @tparam IsColumnA (bool) Whether matrix A is column-major.
/// @tparam IsColumnB (bool) Whether matrix B is column-major.
/// @tparam NumWarps (int) The number of warps per uop.
/// @tparam SmemBytes (int) The size of shared memory per uop.
///
template <typename OutDims, typename NCA, typename NCB, typename Shape,
          typename ProblemSize, typename LeadingDims, int InnerLdimA,
          int InnerLdimB, bool IsColumnA, bool IsColumnB, int NumWarps,
          int SmemBytes, typename DataTypeA, typename DataTypeB,
          typename DataTypeC>
DEVICE void matmul(DataTypeC *C, DataTypeA *A, DataTypeB *B, int uop_idx,
                   int smem_per_warp) {
    static_assert(NCA::D2 == 1 && NCA::D3 == 1,
                  "NCA should be two dimensional.");
    static_assert(NCB::D2 == 1 && NCB::D3 == 1,
                  "NCB should be two dimensional.");
    static_assert(Shape::D3 == 1, "Shape should be three dimensional.");
    static_assert(ProblemSize::D3 == 1,
                  "ProblemSize should be three dimensional.");

    // N dimension of C is max(N dimension of A, N dimension of B)
    constexpr int NC = (NCA::D0 > NCB::D0) ? NCA::D0 : NCB::D0;
    // C dimension of C is max(C dimension of A, C dimension of B)
    constexpr int CC = (NCA::D1 > NCB::D1) ? NCA::D1 : NCB::D1;

    using OutShape = Vec<NC, CC, ProblemSize::D0, ProblemSize::D1>;
    using UnitOutDims = Vec<1, 1, Shape::D0, Shape::D1>;
    using UnitOp = UnitOp<OutDims, OutShape, UnitOutDims, NumWarps, SmemBytes>;

    constexpr int LeadingDimA = LeadingDims::D0;
    constexpr int LeadingDimB = LeadingDims::D3;
    constexpr int LeadingDimC = LeadingDims::D1;
    constexpr int ProblemSizeM = ProblemSize::D0;
    constexpr int ProblemSizeN = ProblemSize::D1;
    constexpr int ProblemSizeK = ProblemSize::D2;
    constexpr int TileSizeM = Shape::D0;
    constexpr int TileSizeN = Shape::D1;
    constexpr int TileSizeK = Shape::D2;

    constexpr DimType SizeA = math::mul<OutDims::H, InnerLdimA>::value;
    constexpr DimType SizeB = math::mul<OutDims::W, InnerLdimB>::value;
    constexpr DimType SizeC = math::mul<OutDims::H, OutDims::W>::value;
    static_assert(SizeA >= 0, "");
    static_assert(SizeB >= 0, "");
    static_assert(SizeC >= 0, "");

    int un = UnitOp::uop_idx_n(uop_idx);
    int uc = UnitOp::uop_idx_c(uop_idx);

    // Broadcasting
    DataTypeA *pA;
    DataTypeB *pB;
    DataTypeC *pC = &C[un * math::mul<CC, SizeC>::value + uc * SizeC];
    if constexpr (NCA::D0 == 1 && NCA::D1 == 1) {
        pA = A;
    } else if constexpr (NCA::D0 == 1) {
        pA = &A[uc * SizeA];
    } else if constexpr (NCA::D1 == 1) {
        pA = &A[un * SizeA];
    } else {
        pA = &A[un * math::mul<CC, SizeA>::value + uc * SizeA];
    }
    if constexpr (NCB::D0 == 1 && NCB::D1 == 1) {
        pB = B;
    } else if constexpr (NCB::D0 == 1) {
        pB = &B[uc * SizeB];
    } else if constexpr (NCB::D1 == 1) {
        pB = &B[un * SizeB];
    } else {
        pB = &B[un * math::mul<CC, SizeB>::value + uc * SizeB];
    }

#if defined(ARK_TARGET_CUDA_ARCH)
    gemm_cutlass<DataTypeA, LeadingDimA, IsColumnA, DataTypeB, LeadingDimB,
                 IsColumnB, DataTypeC, LeadingDimC, ProblemSizeM, ProblemSizeN,
                 ProblemSizeK, TileSizeM, TileSizeN, TileSizeK, UnitOp>(
        pC, pA, pB, uop_idx, smem_per_warp);
#elif defined(ARK_TARGET_ROCM_ARCH)
    gemm_ck<DataTypeA, LeadingDimA, IsColumnA, DataTypeB, LeadingDimB,
            IsColumnB, DataTypeC, LeadingDimC, ProblemSizeM, ProblemSizeN,
            ProblemSizeK, TileSizeM, TileSizeN, TileSizeK, UnitOp>(
        pC, pA, pB, uop_idx, smem_per_warp);
#endif
}

}  // namespace ark

#endif  // ARK_KERNELS_MATMUL_H_
