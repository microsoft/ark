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
    DefaultBroadcast1<InDims, InShape, InDataType, OutDims, OutShape,
                      OutDataType, Copy, UnitOutDims, NumWarps,
                      SmemBytes>::run(out, in, uop_idx);
}

}  // namespace ark

#endif  // ARK_KERNELS_COPY_H_
