// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef ARK_KERNELS_MATMUL_H_
#define ARK_KERNELS_MATMUL_H_

#include "gemm.h"

namespace ark {

// Matrix multiplication. Reuse GEMM kernels. Row-major.
template <typename NCA, typename NCB, typename Shape, typename ProblemSize,
          typename LeadingDims, bool IsColumnA, bool IsColumnB, bool IsRelu, int ThreadsNum, int SmemBytes>
DEVICE void matmul(ark::half *C, ark::half *A, ark::half *B, int tx, int ty,
                   int tz)
{
    // 0x3c00 represents constant 1.0 in half-precision floating point format.
    gemm<NCA, NCB, Shape, ProblemSize, LeadingDims, IsColumnA, IsColumnB, IsRelu, ThreadsNum, SmemBytes>(C, A, B, ark::half::bitcast(0x3c00), ark::half::bitcast(0x0), tz / OutShape::C, tz % OutShape::C, ty, tx);
}

// /* Fused matrix multiplication and scale kernel. */
// template <int M, int N, int K, bool TA, bool TB, int BcastType, bool IsRelu,
//           int ThreadsNum, int SmemBytes, int TDimM, int TDimN, int TDimK>
// DEVICE void matmul(
//    half *C, half *A, half *B, half scale, int tx, int ty, int tz)
// {
//     constexpr int BT = BcastType == 0 ? 0 : BcastType == 1 ? 2 : 1;
//     // 0x3c00 represents constant 1.0 in half-precision floating point
//     // format.
//     gemm<N, M, K, TB, TA, BT, IsRelu, ThreadsNum, SmemBytes, TDimN,
//     TDimM, TDimK>(
//         C, B, A, half{scale}, half::bitcast(0x0), ty, tx, tz);
// }

} // namespace ark

#endif // ARK_KERNELS_MATMUL_H_
