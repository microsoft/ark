// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef ARK_KERNELS_ARITHMETIC_H_
#define ARK_KERNELS_ARITHMETIC_H_

#include "common/broadcast.h"

namespace ark {

template <typename In0Dims, typename In0Shape, typename In1Dims,
          typename In1Shape, typename OutDims, typename OutShape,
          typename UnitOutDims, int NumWarps, int SmemBytes,
          typename In0DataType, typename In1DataType, typename OutDataType>
DEVICE void add(OutDataType *c, const In0DataType *a, const In1DataType *b,
                int uop_idx, int) {
    DefaultBroadcast2<In0Dims, In0Shape, In0DataType, In1Dims, In1Shape,
                      In1DataType, OutDims, OutShape, OutDataType, type::Add,
                      UnitOutDims, NumWarps, SmemBytes>::run(c, a, b, uop_idx);
}

template <typename In0Dims, typename In0Shape, typename In1Dims,
          typename In1Shape, typename OutDims, typename OutShape,
          typename UnitOutDims, int NumWarps, int SmemBytes,
          typename In0DataType, typename In1DataType, typename OutDataType>
DEVICE void sub(OutDataType *c, const In0DataType *a, const In1DataType *b,
                int uop_idx, int) {
    DefaultBroadcast2<In0Dims, In0Shape, In0DataType, In1Dims, In1Shape,
                      In1DataType, OutDims, OutShape, OutDataType, type::Sub,
                      UnitOutDims, NumWarps, SmemBytes>::run(c, a, b, uop_idx);
}

template <typename In0Dims, typename In0Shape, typename In1Dims,
          typename In1Shape, typename OutDims, typename OutShape,
          typename UnitOutDims, int NumWarps, int SmemBytes,
          typename In0DataType, typename In1DataType, typename OutDataType>
DEVICE void mul(OutDataType *c, const In0DataType *a, const In1DataType *b,
                int uop_idx, int) {
    DefaultBroadcast2<In0Dims, In0Shape, In0DataType, In1Dims, In1Shape,
                      In1DataType, OutDims, OutShape, OutDataType, type::Mul,
                      UnitOutDims, NumWarps, SmemBytes>::run(c, a, b, uop_idx);
}

template <typename In0Dims, typename In0Shape, typename In1Dims,
          typename In1Shape, typename OutDims, typename OutShape,
          typename UnitOutDims, int NumWarps, int SmemBytes,
          typename In0DataType, typename In1DataType, typename OutDataType>
DEVICE void div(OutDataType *c, const In0DataType *a, const In1DataType *b,
                int uop_idx, int) {
    DefaultBroadcast2<In0Dims, In0Shape, In0DataType, In1Dims, In1Shape,
                      In1DataType, OutDims, OutShape, OutDataType, type::Div,
                      UnitOutDims, NumWarps, SmemBytes>::run(c, a, b, uop_idx);
}

}  // namespace ark

#endif  // ARK_KERNELS_ARITHMETIC_H_
