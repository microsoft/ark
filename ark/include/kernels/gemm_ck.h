// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef ARK_KERNELS_GEMM_CK_H_
#define ARK_KERNELS_GEMM_CK_H_

#include "ck/ck.hpp"

namespace ark {

/// Row-major GeMM.
template <typename OutDims, typename NCA, typename NCB, typename Shape,
          typename ProblemSize, typename LeadingDims, int InnerLdimA,
          int InnerLdimB, bool IsColumnA, bool IsColumnB, int NumWarps,
          int SmemBytes, typename DataTypeA, typename DataTypeB,
          typename DataTypeC, typename AccumulateType>
DEVICE void gemm_ck(DataTypeC *C, DataTypeA *A, DataTypeB *B, int uop_idx,
                    int smem_per_warp) {}

}  // namespace ark

#endif  // ARK_KERNELS_GEMM_CK_H_
