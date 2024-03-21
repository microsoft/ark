// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef ARK_TENSOR_H_
#define ARK_TENSOR_H_

#include "ark/dims.h"

namespace ark {

/// Calculate new ldims and offs based on the original shape, ldims, offs, and
/// the new shape.
bool tensor_reshape_helper(const Dims &shape, const Dims &ldims,
                           const Dims &offs, const Dims &new_shape,
                           Dims &new_ldims, Dims &new_offs);

}  // namespace ark

#endif  // ARK_TENSOR_H_
