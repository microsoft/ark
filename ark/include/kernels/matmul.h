// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef ARK_KERNELS_MATMUL_H_
#define ARK_KERNELS_MATMUL_H_

#if defined(ARK_TARGET_CUDA_ARCH)
#include "gemm_cutlass.h"
#elif defined(ARK_TARGET_ROCM_ARCH)
#include "gemm_ck.h"
#endif

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
#if defined(ARK_TARGET_CUDA_ARCH)
    gemm_cutlass<OutDims, NCA, NCB, Shape, ProblemSize, LeadingDims, InnerLdimA,
                 InnerLdimB, IsColumnA, IsColumnB, NumWarps, SmemBytes,
                 DataTypeA, DataTypeB, DataTypeC, DataTypeC>(C, A, B, uop_idx,
                                                             smem_per_warp);
#elif defined(ARK_TARGET_ROCM_ARCH)
    gemm_ck<OutDims, NCA, NCB, Shape, ProblemSize, LeadingDims, InnerLdimA,
            InnerLdimB, IsColumnA, IsColumnB, NumWarps, SmemBytes, DataTypeA,
            DataTypeB, DataTypeC, DataTypeC>(C, A, B, uop_idx, smem_per_warp);
#endif
}

template <typename OutDims, typename NCA, typename NCB, typename Shape,
          typename ProblemSize, typename LeadingDims, int InnerLdimA,
          int InnerLdimB, bool IsColumnA, bool IsColumnB, int NumWarps,
          int SmemBytes>
DEVICE void matmul(bf16 *C, bf16 *A, bf16 *B, int uop_idx, int smem_per_warp) {
#if defined(ARK_TARGET_CUDA_ARCH)
    gemm_cutlass<OutDims, NCA, NCB, Shape, ProblemSize, LeadingDims, InnerLdimA,
                 InnerLdimB, IsColumnA, IsColumnB, NumWarps, SmemBytes, bf16,
                 bf16, bf16, float>(C, A, B, uop_idx, smem_per_warp);
#elif defined(ARK_TARGET_ROCM_ARCH)
    gemm_ck<OutDims, NCA, NCB, Shape, ProblemSize, LeadingDims, InnerLdimA,
            InnerLdimB, IsColumnA, IsColumnB, NumWarps, SmemBytes, bf16, bf16,
            bf16, float>(C, A, B, uop_idx, smem_per_warp);
#endif
}

}  // namespace ark

#endif  // ARK_KERNELS_MATMUL_H_
