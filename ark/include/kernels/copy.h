// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef ARK_KERNELS_COPY_H_
#define ARK_KERNELS_COPY_H_

#include "broadcast.h"
#include "type_intrinsics.h"

namespace ark {

struct Copy {
    template <typename DataType>
    static DEVICE DataType compute(DataType input) {
        return input;
    }
};

template <typename InDims, typename InShape, typename OutDims,
          typename OutShape, typename UnitOutDims, int NumWarps, int SmemBytes,
          typename InDataType, typename OutDataType>
DEVICE void copy(OutDataType *out, InDataType *in, int uop_idx,
                 int smem_per_warp) {
    constexpr int NelemPerThread =
        (sizeof(OutDataType) <= 2 && UnitOutDims::W % 8 == 0)
            ? 8
            : (UnitOutDims::W % 4 == 0) ? 4 : (UnitOutDims::W % 2 == 0) ? 2 : 1;
    Broadcast1<InDims, InShape, OutDims, OutShape, UnitOutDims, NumWarps,
               SmemBytes,
               Broadcast1Intrinsic<Copy, InShape, InDataType, OutDataType,
                                   NelemPerThread>>::run(out, in, uop_idx);
}

}  // namespace ark

#endif  // ARK_KERNELS_COPY_H_
