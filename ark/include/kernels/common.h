// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef ARK_KERNELS_COMMON_H_
#define ARK_KERNELS_COMMON_H_

#if defined(__NVCC__) || (defined(__clang__) && defined(__CUDA__)) ||          \
    defined(__CUDACC_RTC__)
#define DEVICE __forceinline__ __device__
#else
#define DEVICE inline
#endif

#endif // ARK_KERNELS_COMMON_H_
