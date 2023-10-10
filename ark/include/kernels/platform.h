// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef ARK_KERNELS_PLATFORM_H_
#define ARK_KERNELS_PLATFORM_H_

#include <cfloat>
#include <limits>

#include "cutlass/platform/platform.h"
#include "device.h"
#include "half.h"

namespace platform {

template <>
struct numeric_limits<float> {
    static DEVICE float lowest() { return -FLT_MAX; }
};

}  // namespace platform

#endif  // ARK_KERNELS_PLATFORM_H_
