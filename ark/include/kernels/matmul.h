// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef ARK_KERNELS_MATMUL_H_
#define ARK_KERNELS_MATMUL_H_

#include "gemm.h"

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
/// @tparam NumThreads (int) The number of threads per uop.
/// @tparam SmemBytes (int) The size of shared memory per uop.
///
template <typename OutDims, typename NCA, typename NCB, typename Shape,
          typename ProblemSize, typename LeadingDims, int InnerLdimA,
          int InnerLdimB, bool IsColumnA, bool IsColumnB, int NumThreads,
          int SmemBytes, typename DataTypeA, typename DataTypeB,
          typename DataTypeC>
DEVICE void matmul(DataTypeC *C, DataTypeA *A, DataTypeB *B, int uop_idx,
                   int smem_per_warp) {
    gemm<OutDims, NCA, NCB, Shape, ProblemSize, LeadingDims, InnerLdimA,
         InnerLdimB, IsColumnA, IsColumnB, NumThreads, SmemBytes, DataTypeA,
         DataTypeB, DataTypeC, DataTypeC>(C, A, B, uop_idx, smem_per_warp);
}

template <typename OutDims, typename NCA, typename NCB, typename Shape,
          typename ProblemSize, typename LeadingDims, int InnerLdimA,
          int InnerLdimB, bool IsColumnA, bool IsColumnB, int NumThreads,
          int SmemBytes>
DEVICE void matmul(bfloat16 *C, bfloat16 *A, bfloat16 *B, int uop_idx,
                   int smem_per_warp) {
    gemm<OutDims, NCA, NCB, Shape, ProblemSize, LeadingDims, InnerLdimA,
         InnerLdimB, IsColumnA, IsColumnB, NumThreads, SmemBytes, bfloat16,
         bfloat16, bfloat16, float>(C, A, B, uop_idx, smem_per_warp);
}

}  // namespace ark

#endif  // ARK_KERNELS_MATMUL_H_
