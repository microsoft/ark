// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.
#include <algorithm>
#include <ctime>
#include <random>
#include <sys/syscall.h>
#include <unistd.h>
#define gettid() syscall(SYS_gettid)

#include "ark/random.h"

namespace ark {

// Initialize the random number generator.
void srand(int seed)
{
    if (seed == 0) {
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

// Generate a random alpha-numeric string.
const std::string rand_anum(size_t len)
{
    auto randchar = []() -> char {
        const char charset[] = "0123456789"
                               "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                               "abcdefghijklmnopqrstuvwxyz";
        const size_t max_index = sizeof(charset) - 1;
        return charset[rand() % max_index];
    };
    std::string str(len, 0);
    std::generate_n(str.begin(), len, randchar);
    return str;
}

} // namespace ark
