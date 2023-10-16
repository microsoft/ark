// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef ARK_KERNELS_ARCH_H_
#define ARK_KERNELS_ARCH_H_

#include "device.h"
#include "static_math.h"

namespace ark {

struct Arch {
#if defined(ARK_TARGET_ROCM_CUDA)
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
