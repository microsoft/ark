// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef ARK_KERNELS_BFLOAT16_H_
#define ARK_KERNELS_BFLOAT16_H_

// clang-format off
#include "cutlass/numeric_types.h"
#include "cutlass/bfloat16.h"
// clang-format on

namespace ark {
using bfloat16 = cutlass::bfloat16_t;
}  // namespace ark

#endif  // ARK_KERNELS_BFLOAT16_H_
