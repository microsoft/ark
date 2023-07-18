// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include <ctime>
#include <random>
#include <sys/syscall.h>
#include <unistd.h>
#define gettid() syscall(SYS_gettid)

#include "include/ark.h"

namespace ark {

// Initialize the random number generator.
void srand(int seed)
{
    if (seed == -1) {
        ::srand(time(0) + getpid() + gettid());
    } else {
        ::srand(seed);
    }
}

// Generate a random integer.
int rand()
{
    return ::rand();
}

} // namespace ark
