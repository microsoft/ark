// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef ARK_VERSION_HPP
#define ARK_VERSION_HPP

#include <string>

#define ARK_MAJOR 0
#define ARK_MINOR 5
#define ARK_PATCH 0
#define ARK_VERSION (ARK_MAJOR * 10000 + ARK_MINOR * 100 + ARK_PATCH)

namespace ark {

/// Return a version string.
std::string version();

}  // namespace ark

#endif  // ARK_VERSION_HPP
