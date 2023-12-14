// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef ARK_KERNELS_SHFL_H_
#define ARK_KERNELS_SHFL_H_

namespace ark {

#if defined(ARK_TARGET_CUDA_ARCH)
#define SHFL_XOR(var, lane_mask, width) \
    __shfl_xor_sync(0xffffffff, var, lane_mask, width)
#elif defined(ARK_TARGET_ROCM_ARCH)
#define SHFL_XOR(var, lane_mask, width) __shfl_xor(var, lane_mask, width)
#endif

}  // namespace ark

#endif  // ARK_KERNELS_SHFL_H_
