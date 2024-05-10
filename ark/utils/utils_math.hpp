// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef ARK_UTILS_MATH_HPP_
#define ARK_UTILS_MATH_HPP_

#include <cstddef>

namespace ark {
namespace math {

size_t div_up(size_t x, size_t div);
size_t pad(size_t x, size_t u);
bool is_pow2(size_t x);
unsigned int ilog2(unsigned int x);
size_t gcd(size_t a, size_t b);
size_t lcm(size_t a, size_t b);

}  // namespace math
}  // namespace ark

#endif  // ARK_UTILS_MATH_HPP_
