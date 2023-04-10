// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.
#ifndef ARK_RANDOM_H_
#define ARK_RANDOM_H_

#include <string>

namespace ark {

void srand(int seed = 0);
int rand();

const std::string rand_anum(size_t len);

} // namespace ark

#endif // ARK_RANDOM_H_