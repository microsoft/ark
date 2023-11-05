// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef ARK_KERNELS_SHFL_H_
#define ARK_KERNELS_SHFL_H_

namespace ark {

#define SHFL_XOR(var, lane_mask, width) \
    __shfl_xor_sync(0xffffffff, var, lane_mask, width)

}  // namespace ark

#endif  // ARK_KERNELS_SHFL_H_
