// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef ARK_KERNELS_PLATFORM_H_
#define ARK_KERNELS_PLATFORM_H_

#include "common.h"
#include "half.h"
#include <cfloat>
#include <limits>
namespace platform {
template <> struct numeric_limits<float>
{
    static DEVICE float lowest()
    {
        return -FLT_MAX;
    }
};
} // namespace platform

#endif // ARK_KERNELS_PLATFORM_H_
