// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef ARK_KERNELS_HALF_H_
#define ARK_KERNELS_HALF_H_

// clang-format off
#include "cutlass/numeric_types.h"
#include "cutlass/half.h"
// clang-format on

namespace ark {
using half = cutlass::half_t;
}  // namespace ark

#endif  // ARK_KERNELS_HALF_H_
