// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef ARK_KERNELS_DEVICE_H_
#define ARK_KERNELS_DEVICE_H_

#if defined(ARK_TARGET_CUDA_ARCH)
#define DEVICE __forceinline__ __device__
#else
#define DEVICE __device__ inline
#endif

#endif  // ARK_KERNELS_DEVICE_H_
