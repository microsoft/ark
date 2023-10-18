// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef ARK_KERNELS_GEMM_H_
#define ARK_KERNELS_GEMM_H_

#include <utility>

#if defined(ARK_TARGET_CUDA_ARCH)
#include "gemm_cutlass.h"
#elif defined(ARK_TARGET_ROCM_ARCH)
#include "gemm_ck.h"
#endif

namespace ark {

/// Row-major GeMM.
template <typename... Args>
DEVICE void gemm(Args... args) {
#if defined(ARK_TARGET_CUDA_ARCH)
    gemm_cutlass(std::forward<Args>(args)...);
#elif defined(ARK_TARGET_ROCM_ARCH)
    gemm_ck(std::forward<Args>(args)...);
#endif
}

}  // namespace ark

#endif  // ARK_KERNELS_GEMM_H_
