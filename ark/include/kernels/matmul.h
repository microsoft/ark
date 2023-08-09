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
/// @tparam IsColumnA (bool) Whether matrix A is column-major.
/// @tparam IsColumnB (bool) Whether matrix B is column-major.
/// @tparam NumThreads (int) The number of threads per uop.
/// @tparam SmemBytes (int) The size of shared memory per uop.
///
template <typename OutDims, typename NCA, typename NCB, typename Shape,
          typename ProblemSize, typename LeadingDims, bool IsColumnA,
          bool IsColumnB, int NumThreads, int SmemBytes>
DEVICE void matmul(float *C, float *A, float *B, int uop_idx, int smem_per_warp)
{
    gemm<OutDims, NCA, NCB, Shape, ProblemSize, LeadingDims, IsColumnA,
         IsColumnB, NumThreads, SmemBytes, float, float, float, float>(
        C, A, B, uop_idx, smem_per_warp);
}

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
/// @tparam IsColumnA (bool) Whether matrix A is column-major.
/// @tparam IsColumnB (bool) Whether matrix B is column-major.
/// @tparam NumThreads (int) The number of threads per uop.
/// @tparam SmemBytes (int) The size of shared memory per uop.
///
template <typename OutDims, typename NCA, typename NCB, typename Shape,
          typename ProblemSize, typename LeadingDims, bool IsColumnA,
          bool IsColumnB, int NumThreads, int SmemBytes>
DEVICE void matmul(half *C, half *A, half *B, int uop_idx, int smem_per_warp)
{
    gemm<OutDims, NCA, NCB, Shape, ProblemSize, LeadingDims, IsColumnA,
         IsColumnB, NumThreads, SmemBytes, half, half, half, half>(
        C, A, B, uop_idx, smem_per_warp);
}

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
/// @tparam IsColumnA (bool) Whether matrix A is column-major.
/// @tparam IsColumnB (bool) Whether matrix B is column-major.
/// @tparam NumThreads (int) The number of threads per uop.
/// @tparam SmemBytes (int) The size of shared memory per uop.
///
template <typename OutDims, typename NCA, typename NCB, typename Shape,
          typename ProblemSize, typename LeadingDims, bool IsColumnA,
          bool IsColumnB, int NumThreads, int SmemBytes>
DEVICE void matmul(int8_t *C, int8_t *A, int8_t *B, int uop_idx,
                   int smem_per_warp)
{
    gemm<OutDims, NCA, NCB, Shape, ProblemSize, LeadingDims, IsColumnA,
         IsColumnB, NumThreads, SmemBytes, int8_t, int8_t, int8_t, int32_t>(
        C, A, B, uop_idx, smem_per_warp);
}

} // namespace ark

#endif // ARK_KERNELS_MATMUL_H_
