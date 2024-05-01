// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef ARK_H
#define ARK_H

#include <string>

#define ARK_MAJOR 0
#define ARK_MINOR 5
#define ARK_PATCH 0
#define ARK_VERSION (ARK_MAJOR * 10000 + ARK_MINOR * 100 + ARK_PATCH)

// #include "ark/error.hpp"
// #include "ark/executor.hpp"
#include "ark/model.hpp"

namespace ark {

/// Return a version string.
std::string version();

// set random seed
void srand(int seed = -1);

// get random number
int rand();

/// Initialize the ARK runtime.
///
/// This function should be called by the user before any other functions are
/// called. It is safe to call this function multiple times.
void init();

}  // namespace ark

#endif  // ARK_H
