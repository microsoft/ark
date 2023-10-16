// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef ARK_KERNELS_ARCH_H_
#define ARK_KERNELS_ARCH_H_

#if defined(ARK_TARGET_ROCM_ARCH)
#include <hip/hip_runtime.h>
#endif  // ARK_TARGET_ROCM_ARCH

#include "device.h"
#include "static_math.h"

namespace ark {

struct Arch {
#if defined(ARK_TARGET_CUDA_ARCH)
    static const int ThreadsPerWarp = 32;
#elif defined(ARK_TARGET_ROCM_ARCH)
    static const int ThreadsPerWarp = 64;
#endif
};

DEVICE int warp_id() {
    return threadIdx.x >> math::log2_up<Arch::ThreadsPerWarp>::value;
}

}  // namespace ark

#endif  // ARK_KERNELS_ARCH_H_
