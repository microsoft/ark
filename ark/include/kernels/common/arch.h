// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef ARK_KERNELS_ARCH_H_
#define ARK_KERNELS_ARCH_H_

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

#if defined(ARK_TARGET_CUDA_ARCH)
#define ARCH_ALIAS_TYPE(alias, cuda_type, hip_type) typedef cuda_type alias;
#elif defined(ARK_TARGET_ROCM_ARCH)
#define ARCH_ALIAS_TYPE(alias, cuda_type, hip_type) typedef hip_type alias;
#endif

#if defined(ARK_TARGET_CUDA_ARCH)
#define ARCH_ALIAS_FUNC(alias, cuda_func, hip_func)    \
    template <typename... Args>                        \
    inline auto alias(Args &&... args) {               \
        return cuda_func(std::forward<Args>(args)...); \
    }
#elif defined(ARK_TARGET_ROCM_ARCH)
#define ARCH_ALIAS_FUNC(alias, cuda_func, hip_func)   \
    template <typename... Args>                       \
    inline auto alias(Args &&... args) {              \
        return hip_func(std::forward<Args>(args)...); \
    }
#endif

#endif  // ARK_KERNELS_ARCH_H_
