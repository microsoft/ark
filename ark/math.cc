// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "math.h"

#include "include/ark.h"
#include "logging.h"

namespace ark {
namespace math {

// Calculate the ceiling of x / div.
size_t div_up(size_t x, size_t div) {
    if (div == 0) {
        ERR(InvalidUsageError, "division by zero");
    }
    if (x == 0) {
        return 0;
    }
    return 1 + ((x - 1) / div);
}

// Calculate the minimum multiple of u that is greater than or equal to x.
size_t pad(size_t x, size_t u) { return div_up(x, u) * u; }

// Return true if x is a power of 2.
bool is_pow2(size_t x) {
    if (x == 0) {
        return false;
    }
    return (x & (x - 1)) == 0;
}

// Return the log base 2 of x. x must be a power of 2.
unsigned int ilog2(unsigned int x) {
    if (x == 0) {
        ERR(InvalidUsageError, "log of zero is undefined");
    }
    return (sizeof(unsigned int) * 8) - __builtin_clz(x) - 1;
}

// Greatest Common Divisor.
size_t gcd(size_t a, size_t b) {
    if (a == 0) {
        return b;
    }
    if (b == 0) {
        return a;
    }
    while (b != 0) {
        size_t t = b;
        b = a % b;
        a = t;
    }
    return a;
}

// Least Common Multiple.
size_t lcm(size_t a, size_t b) { return a / gcd(a, b) * b; }

}  // namespace math
}  // namespace ark
