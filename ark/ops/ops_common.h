// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef ARK_OPS_COMMON_H_
#define ARK_OPS_COMMON_H_

#include <map>
#include <ostream>
#include <vector>

#include "include/ark.h"

namespace ark {

/// Return the output shape of broadcasting between two shapes.
/// Follow NumPy rules.
/// https://numpy.org/doc/stable/user/basics.broadcasting.html
/// @param dims1 The first shape.
/// @param dims2 The second shape.
Dims broadcast(const Dims &dims1, const Dims &dims2);

}  // namespace ark

#endif  // ARK_OPS_COMMON_H_
