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

/// Generate a random value.
template <typename T>
T rand(float min_val, float max_val) {
    int mid = std::numeric_limits<int>::max() / 2;
    return T((ark::rand() - mid) / (float)mid * (max_val - min_val) + min_val);
}

}  // namespace ark

#endif  // ARK_RANDOM_HPP
