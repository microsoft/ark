// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef ARK_KERNELS_MATH_FUNCTIONS_H_
#define ARK_KERNELS_MATH_FUNCTIONS_H_

#include "common.h"

namespace ark {

template <typename InDims, typename InShape, typename OutDims,
          typename OutShape, typename UnitOutDims, int NumThreads,
          int SmemBytes, typename InDataType, typename OutDataType>
DEVICE void exp(OutDataType *out, const InDataType *in, int uop_idx, int) {
    constexpr int NelemPerThread =
        (sizeof(OutDataType) <= 2 && UnitOutDims::W % 8 == 0)
            ? 8
            : (UnitOutDims::W % 4 == 0) ? 4 : (UnitOutDims::W % 2 == 0) ? 2 : 1;
    Broadcast1<InDims, InShape, OutDims, OutShape, UnitOutDims, NumThreads,
               SmemBytes,
               Broadcast1Intrinsic<type::Exp, InShape, InDataType, OutDataType,
                                   NelemPerThread>>::run(out, in, uop_idx);
}

template <typename InDims, typename InShape, typename OutDims,
          typename OutShape, typename UnitOutDims, int NumThreads,
          int SmemBytes, typename InDataType, typename OutDataType>
DEVICE void sqrt(OutDataType *out, const InDataType *in, int uop_idx, int) {
    constexpr int NelemPerThread =
        (sizeof(OutDataType) <= 2 && UnitOutDims::W % 8 == 0)
            ? 8
            : (UnitOutDims::W % 4 == 0) ? 4 : (UnitOutDims::W % 2 == 0) ? 2 : 1;
    Broadcast1<InDims, InShape, OutDims, OutShape, UnitOutDims, NumThreads,
               SmemBytes,
               Broadcast1Intrinsic<type::Sqrt, InShape, InDataType, OutDataType,
                                   NelemPerThread>>::run(out, in, uop_idx);
}

}  // namespace ark

#endif  // ARK_KERNELS_MATH_FUNCTIONS_H_
