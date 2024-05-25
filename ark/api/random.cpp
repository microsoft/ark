// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "ark/random.hpp"

#include <random>

namespace ark {

// Initialize the random number generator.
void srand(int seed) { ::srand(seed); }

// Generate a random integer.
int rand() { return ::rand(); }

}  // namespace ark
