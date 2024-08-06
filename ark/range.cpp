// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "range.hpp"

namespace ark {

std::ostream& operator<<(std::ostream& os, const Range<int>& range) {
    if (range.step() == 1) {
        os << "(" << *range.begin() << ", " << *range.end() << ")";
    } else {
        os << "(" << *range.begin() << ", " << *range.end() << ", "
           << range.step() << ")";
    }
    return os;
}

}  // namespace ark
