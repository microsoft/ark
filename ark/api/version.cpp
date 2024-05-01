// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "ark/version.hpp"

#include <sstream>
#include <string>

namespace ark {

std::string version() {
    std::stringstream ss;
    ss << ARK_MAJOR << "." << ARK_MINOR << "." << ARK_PATCH;
    return ss.str();
}

}  // namespace ark
