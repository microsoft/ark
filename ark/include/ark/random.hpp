// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef ARK_RANDOM_HPP
#define ARK_RANDOM_HPP

namespace ark {

// set random seed
void srand(int seed = -1);

// get random number
int rand();

/// Generate a random value.
template <typename T>
T rand(float min_val, float max_val) {
    int mid = RAND_MAX / 2;
    return T((ark::rand() - mid) / (float)mid * (max_val - min_val) + min_val);
}

}  // namespace ark

#endif  // ARK_RANDOM_HPP
