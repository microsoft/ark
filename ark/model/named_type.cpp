// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "named_type.hpp"

namespace ark {

bool operator==(const NamedT &lhs, const NamedT &rhs) {
    return lhs.type_name() == rhs.type_name();
}

}  // namespace ark
