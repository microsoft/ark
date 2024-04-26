// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef ARK_RANDOM_H_
#define ARK_RANDOM_H_

namespace ark {

/// Generate a random alpha-numeric string.
/// @param len Length of the string
/// @return A random alpha-numeric string
std::string rand_anum(size_t len);

}  // namespace ark

#endif  // ARK_RANDOM_H_
