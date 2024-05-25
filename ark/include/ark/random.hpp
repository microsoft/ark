// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef ARK_RANDOM_HPP
#define ARK_RANDOM_HPP

#include <limits>

namespace ark {

// set random seed
void srand(int seed = -1);

// get random number
int rand();

}  // namespace ark

#endif  // ARK_RANDOM_HPP
