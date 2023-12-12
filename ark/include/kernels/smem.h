// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef ARK_KERNELS_SMEM_H_
#define ARK_KERNELS_SMEM_H_

#include "arch.h"
#include "device.h"
#include "static_math.h"

extern __shared__ int _ARK_SMEM[];

// should be multiple of 128 and equal to or larger than sync::WarpGroupState
#define ARK_SMEM_RESERVED_BYTES 128

namespace ark {

template <typename T, int NumWarps>
struct SharedMemory {
    static DEVICE int smem_base_offset(int smem_per_warp) {
        // The smallest warp ID in the uop.
        int least_warp_id = math::gm<NumWarps>(warp_id());
        return math::div<sizeof(int)>(least_warp_id * smem_per_warp +
                                      ARK_SMEM_RESERVED_BYTES);
    }

    static DEVICE T *get(int smem_per_warp) {
        return (T *)&_ARK_SMEM[smem_base_offset(smem_per_warp)];
    }
};

}  // namespace ark

#endif  // ARK_KERNELS_SMEM_H_
