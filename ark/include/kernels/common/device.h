// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef ARK_KERNELS_DEVICE_H_
#define ARK_KERNELS_DEVICE_H_

#if defined(ARK_TARGET_CUDA_ARCH) && defined(ARK_TARGET_ROCM_ARCH)
static_assert(false, "Multiple GPU architectures");
#endif  // defined(ARK_TARGET_CUDA_ARCH) && defined(ARK_TARGET_ROCM_ARCH)

#if defined(ARK_TARGET_ROCM_ARCH)
#include <hip/hip_runtime.h>
#endif  // !defined(ARK_TARGET_CUDA_ARCH)

#if !defined(ARK_TARGET_CUDA_ARCH) && !defined(ARK_TARGET_ROCM_ARCH)
static_assert(false, "Unknown GPU architecture");
#define ARK_TARGET_CUDA_ARCH 800  // Dummy define
#include <cuda_runtime.h>         // Dummy include
#endif  // !defined(ARK_TARGET_CUDA_ARCH) && !defined(ARK_TARGET_ROCM_ARCH)

#define DEVICE __forceinline__ __device__

#endif  // ARK_KERNELS_DEVICE_H_
