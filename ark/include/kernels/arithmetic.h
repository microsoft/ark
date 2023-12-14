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
                int uop_idx, int smem_per_warp) {
    broadcast2<In0Dims, In0Shape, In1Dims, In1Shape, OutDims, OutShape,
               UnitOutDims, NumWarps, SmemBytes, In0DataType, In1DataType,
               OutDataType, type::Add>(c, a, b, uop_idx, smem_per_warp);
}

template <typename In0Dims, typename In0Shape, typename In1Dims,
          typename In1Shape, typename OutDims, typename OutShape,
          typename UnitOutDims, int NumWarps, int SmemBytes,
          typename In0DataType, typename In1DataType, typename OutDataType>
DEVICE void sub(OutDataType *c, const In0DataType *a, const In1DataType *b,
                int uop_idx, int smem_per_warp) {
    broadcast2<In0Dims, In0Shape, In1Dims, In1Shape, OutDims, OutShape,
               UnitOutDims, NumWarps, SmemBytes, In0DataType, In1DataType,
               OutDataType, type::Sub>(c, a, b, uop_idx, smem_per_warp);
}

template <typename In0Dims, typename In0Shape, typename In1Dims,
          typename In1Shape, typename OutDims, typename OutShape,
          typename UnitOutDims, int NumWarps, int SmemBytes,
          typename In0DataType, typename In1DataType, typename OutDataType>
DEVICE void mul(OutDataType *c, const In0DataType *a, const In1DataType *b,
                int uop_idx, int smem_per_warp) {
    broadcast2<In0Dims, In0Shape, In1Dims, In1Shape, OutDims, OutShape,
               UnitOutDims, NumWarps, SmemBytes, In0DataType, In1DataType,
               OutDataType, type::Mul>(c, a, b, uop_idx, smem_per_warp);
}

template <typename In0Dims, typename In0Shape, typename In1Dims,
          typename In1Shape, typename OutDims, typename OutShape,
          typename UnitOutDims, int NumWarps, int SmemBytes,
          typename In0DataType, typename In1DataType, typename OutDataType>
DEVICE void div(OutDataType *c, const In0DataType *a, const In1DataType *b,
                int uop_idx, int smem_per_warp) {
    broadcast2<In0Dims, In0Shape, In1Dims, In1Shape, OutDims, OutShape,
               UnitOutDims, NumWarps, SmemBytes, In0DataType, In1DataType,
               OutDataType, type::Div>(c, a, b, uop_idx, smem_per_warp);
}

template <typename InDims, typename InShape, typename OutDims,
          typename OutShape, typename UnitOutDims, int NumWarps, int SmemBytes,
          typename InDataType, typename OutDataType>
DEVICE void scale(OutDataType *y, InDataType *x, float val, int uop_idx,
                  int smem_per_warp) {
    InDataType val_cast = type::Cast::compute<InDataType>(val);
    using ValDims = Vec<1, 1, 1, 1>;
    using ValShape = Vec<1, 1, 1, 1>;
    broadcast2<InDims, InShape, ValDims, ValShape, OutDims, OutShape,
               UnitOutDims, NumWarps, SmemBytes, InDataType, InDataType,
               OutDataType, type::Mul>(y, x, &val_cast, uop_idx, smem_per_warp);
}

}  // namespace ark

#endif  // ARK_KERNELS_ARITHMETIC_H_
