// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef ARK_KERNELS_GEMM_CK_H_
#define ARK_KERNELS_GEMM_CK_H_

#include "ck/ck.hpp"

namespace ark {

/// Row-major GeMM.
template <typename DataTypeA, int LeadingDimA, bool IsColumnA,
          typename DataTypeB, int LeadingDimB, bool IsColumnB,
          typename DataTypeC, int LeadingDimC, int ProblemSizeM,
          int ProblemSizeN, int ProblemSizeK, int TileSizeM, int TileSizeN,
          int TileSizeK, typename UnitOp>
DEVICE void gemm_ck(DataTypeC *C, DataTypeA *A, DataTypeB *B, int uop_idx,
                    int smem_per_warp) {}

}  // namespace ark

#endif  // ARK_KERNELS_GEMM_CK_H_
