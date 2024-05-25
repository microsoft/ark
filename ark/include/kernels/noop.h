// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef ARK_KERNELS_NOOP_H_
#define ARK_KERNELS_NOOP_H_

#include "common/device.h"

namespace ark {

DEVICE void noop(int, int) {}

}  // namespace ark

#endif  // ARK_KERNELS_NOOP_H_
