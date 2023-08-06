// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef ARK_KERNELS_ARCH_H_
#define ARK_KERNELS_ARCH_H_

#include "device.h"
#include "static_math.h"

namespace ark {

struct Arch
{
#if (ARK_TARGET_CUDA_ARCH == 60)
    static const int ThreadsPerWarp = 32;
    static const int MaxRegistersPerBlock = 65536;
    static const int MaxSmemBytesPerBlock = 49152;
    static const int MaxRegistersPerThread = 256;
#elif (ARK_TARGET_CUDA_ARCH == 70)
    static const int ThreadsPerWarp = 32;
    static const int MaxRegistersPerBlock = 65536;
    static const int MaxSmemBytesPerBlock = 98304;
    static const int MaxRegistersPerThread = 256;
#elif (ARK_TARGET_CUDA_ARCH == 75)
    static const int ThreadsPerWarp = 32;
    static const int MaxRegistersPerBlock = 65536;
    static const int MaxSmemBytesPerBlock = 65536;
    static const int MaxRegistersPerThread = 256;
#elif (ARK_TARGET_CUDA_ARCH == 80)
    static const int ThreadsPerWarp = 32;
    static const int MaxRegistersPerBlock = 65536;
    static const int MaxSmemBytesPerBlock = 166912;
    static const int MaxRegistersPerThread = 256;
#endif

    static const int ArkMinThreadsPerBlock =
        MaxRegistersPerBlock / MaxRegistersPerThread;
};

DEVICE int warp_id()
{
    return threadIdx.x >> math::log2_up<Arch::ThreadsPerWarp>::value;
}

} // namespace ark

#endif // ARK_KERNELS_ARCH_H_
