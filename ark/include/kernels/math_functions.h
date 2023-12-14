// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef ARK_KERNELS_MATH_FUNCTIONS_H_
#define ARK_KERNELS_MATH_FUNCTIONS_H_

#include "common/broadcast.h"

namespace ark {

template <typename InDims, typename InShape, typename OutDims,
          typename OutShape, typename UnitOutDims, int NumWarps, int SmemBytes,
          typename InDataType, typename OutDataType>
DEVICE void exp(OutDataType *out, const InDataType *in, int uop_idx, int) {
    DefaultBroadcast1<InDims, InShape, InDataType, OutDims, OutShape,
                      OutDataType, type::Exp, UnitOutDims, NumWarps,
                      SmemBytes>::run(out, in, uop_idx);
}

template <typename InDims, typename InShape, typename OutDims,
          typename OutShape, typename UnitOutDims, int NumWarps, int SmemBytes,
          typename InDataType, typename OutDataType>
DEVICE void sqrt(OutDataType *out, const InDataType *in, int uop_idx, int) {
    DefaultBroadcast1<InDims, InShape, InDataType, OutDims, OutShape,
                      OutDataType, type::Sqrt, UnitOutDims, NumWarps,
                      SmemBytes>::run(out, in, uop_idx);
}

template <typename InDims, typename InShape, typename OutDims,
          typename OutShape, typename UnitOutDims, int NumWarps, int SmemBytes,
          typename InDataType, typename OutDataType>
DEVICE void rsqrt(OutDataType *out, const InDataType *in, int uop_idx, int) {
    DefaultBroadcast1<InDims, InShape, InDataType, OutDims, OutShape,
                      OutDataType, type::Rsqrt, UnitOutDims, NumWarps,
                      SmemBytes>::run(out, in, uop_idx);
}

}  // namespace ark

#endif  // ARK_KERNELS_MATH_FUNCTIONS_H_
