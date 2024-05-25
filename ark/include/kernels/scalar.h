// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef ARK_KERNELS_SCALAR_H_
#define ARK_KERNELS_SCALAR_H_

#include "common/broadcast.h"

namespace ark {

template <typename OutDims, typename OutShape, typename UnitOutDims,
          int NumWarps, int SmemBytes, typename OutDataType>
DEVICE void scalar_assign(OutDataType *out, float val, int uop_idx, int) {
    OutDataType val_cast = type::Cast::compute<OutDataType>(val);
    using ValDims = Vec<1, 1, 1, 1>;
    using ValShape = Vec<1, 1, 1, 1>;
    DefaultBroadcast1<ValDims, ValShape, OutDataType, OutDims, OutShape,
                      OutDataType, type::Identity, false, false, UnitOutDims,
                      NumWarps, SmemBytes>::run(out, &val_cast, uop_idx);
}

template <typename InDims, typename InShape, typename OutDims,
          typename OutShape, typename UnitOutDims, int NumWarps, int SmemBytes,
          typename InDataType, typename OutDataType>
DEVICE void scalar_add(OutDataType *y, InDataType *x, float val, int uop_idx,
                       int) {
    InDataType val_cast = type::Cast::compute<InDataType>(val);
    using ValDims = Vec<1, 1, 1, 1>;
    using ValShape = Vec<1, 1, 1, 1>;
    DefaultBroadcast2<InDims, InShape, InDataType, ValDims, ValShape,
                      InDataType, OutDims, OutShape, OutDataType, type::Add,
                      UnitOutDims, NumWarps, SmemBytes>::run(y, x, &val_cast,
                                                             uop_idx);
}

template <typename InDims, typename InShape, typename OutDims,
          typename OutShape, typename UnitOutDims, int NumWarps, int SmemBytes,
          typename InDataType, typename OutDataType>
DEVICE void scalar_mul(OutDataType *y, InDataType *x, float val, int uop_idx,
                       int) {
    InDataType val_cast = type::Cast::compute<InDataType>(val);
    using ValDims = Vec<1, 1, 1, 1>;
    using ValShape = Vec<1, 1, 1, 1>;
    DefaultBroadcast2<InDims, InShape, InDataType, ValDims, ValShape,
                      InDataType, OutDims, OutShape, OutDataType, type::Mul,
                      UnitOutDims, NumWarps, SmemBytes>::run(y, x, &val_cast,
                                                             uop_idx);
}

}  // namespace ark

#endif  // ARK_KERNELS_SCALAR_H_
