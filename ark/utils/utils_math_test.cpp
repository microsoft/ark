// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "utils/utils_math.hpp"

#include "unittest/unittest_utils.h"

ark::unittest::State test_utils_math() {
    UNITTEST_EQ(ark::math::div_up(0, 1), 0);
    UNITTEST_EQ(ark::math::div_up(1, 1), 1);
    UNITTEST_EQ(ark::math::div_up(1, 2), 1);
    UNITTEST_EQ(ark::math::div_up(2, 2), 1);
    UNITTEST_EQ(ark::math::div_up(3, 2), 2);
    UNITTEST_EQ(ark::math::div_up(4, 2), 2);
    UNITTEST_EQ(ark::math::div_up(5, 2), 3);
    UNITTEST_EQ(ark::math::div_up(6, 2), 3);
    UNITTEST_EQ(ark::math::div_up(7, 2), 4);
    UNITTEST_EQ(ark::math::div_up(7, 5), 2);
    UNITTEST_EQ(ark::math::div_up(7, 7), 1);
    UNITTEST_EQ(ark::math::div_up(7, 8), 1);

    UNITTEST_EQ(ark::math::pad(0, 1), 0);
    UNITTEST_EQ(ark::math::pad(1, 1), 1);
    UNITTEST_EQ(ark::math::pad(1, 2), 2);
    UNITTEST_EQ(ark::math::pad(2, 2), 2);
    UNITTEST_EQ(ark::math::pad(3, 2), 4);
    UNITTEST_EQ(ark::math::pad(4, 2), 4);
    UNITTEST_EQ(ark::math::pad(5, 2), 6);
    UNITTEST_EQ(ark::math::pad(6, 2), 6);
    UNITTEST_EQ(ark::math::pad(7, 2), 8);
    UNITTEST_EQ(ark::math::pad(7, 5), 10);
    UNITTEST_EQ(ark::math::pad(7, 7), 7);
    UNITTEST_EQ(ark::math::pad(7, 8), 8);

    UNITTEST_TRUE(ark::math::is_pow2(1));
    UNITTEST_TRUE(ark::math::is_pow2(2));
    UNITTEST_FALSE(ark::math::is_pow2(3));
    UNITTEST_TRUE(ark::math::is_pow2(4));
    UNITTEST_FALSE(ark::math::is_pow2(5));
    UNITTEST_FALSE(ark::math::is_pow2(6));
    UNITTEST_FALSE(ark::math::is_pow2(7));
    UNITTEST_TRUE(ark::math::is_pow2(8));
    UNITTEST_FALSE(ark::math::is_pow2(9));
    UNITTEST_FALSE(ark::math::is_pow2(10));
    UNITTEST_FALSE(ark::math::is_pow2(11));
    UNITTEST_FALSE(ark::math::is_pow2(12));
    UNITTEST_FALSE(ark::math::is_pow2(13));
    UNITTEST_FALSE(ark::math::is_pow2(14));
    UNITTEST_FALSE(ark::math::is_pow2(15));
    UNITTEST_TRUE(ark::math::is_pow2(16));

    UNITTEST_EQ(ark::math::ilog2(1), 0);
    UNITTEST_EQ(ark::math::ilog2(2), 1);
    UNITTEST_EQ(ark::math::ilog2(4), 2);
    UNITTEST_EQ(ark::math::ilog2(8), 3);
    UNITTEST_EQ(ark::math::ilog2(16), 4);
    UNITTEST_EQ(ark::math::ilog2(32), 5);
    UNITTEST_EQ(ark::math::ilog2(64), 6);

    UNITTEST_EQ(ark::math::gcd(1, 1), 1);
    UNITTEST_EQ(ark::math::gcd(1, 2), 1);
    UNITTEST_EQ(ark::math::gcd(2, 1), 1);
    UNITTEST_EQ(ark::math::gcd(2, 2), 2);
    UNITTEST_EQ(ark::math::gcd(2, 3), 1);
    UNITTEST_EQ(ark::math::gcd(3, 2), 1);
    UNITTEST_EQ(ark::math::gcd(3, 3), 3);
    UNITTEST_EQ(ark::math::gcd(3, 4), 1);
    UNITTEST_EQ(ark::math::gcd(4, 3), 1);
    UNITTEST_EQ(ark::math::gcd(4, 4), 4);
    UNITTEST_EQ(ark::math::gcd(4, 5), 1);
    UNITTEST_EQ(ark::math::gcd(5, 4), 1);
    UNITTEST_EQ(ark::math::gcd(5, 5), 5);
    UNITTEST_EQ(ark::math::gcd(5, 6), 1);
    UNITTEST_EQ(ark::math::gcd(6, 5), 1);
    UNITTEST_EQ(ark::math::gcd(6, 6), 6);
    UNITTEST_EQ(ark::math::gcd(32, 58), 2);
    UNITTEST_EQ(ark::math::gcd(58, 32), 2);
    UNITTEST_EQ(ark::math::gcd(72, 78), 6);

    UNITTEST_EQ(ark::math::lcm(1, 1), 1);
    UNITTEST_EQ(ark::math::lcm(1, 2), 2);
    UNITTEST_EQ(ark::math::lcm(2, 1), 2);
    UNITTEST_EQ(ark::math::lcm(2, 2), 2);
    UNITTEST_EQ(ark::math::lcm(2, 3), 6);
    UNITTEST_EQ(ark::math::lcm(3, 2), 6);
    UNITTEST_EQ(ark::math::lcm(3, 3), 3);
    UNITTEST_EQ(ark::math::lcm(3, 4), 12);
    UNITTEST_EQ(ark::math::lcm(4, 3), 12);
    UNITTEST_EQ(ark::math::lcm(4, 4), 4);
    UNITTEST_EQ(ark::math::lcm(4, 5), 20);
    UNITTEST_EQ(ark::math::lcm(5, 4), 20);
    UNITTEST_EQ(ark::math::lcm(5, 5), 5);
    UNITTEST_EQ(ark::math::lcm(5, 6), 30);
    UNITTEST_EQ(ark::math::lcm(6, 5), 30);
    UNITTEST_EQ(ark::math::lcm(6, 6), 6);
    UNITTEST_EQ(ark::math::lcm(32, 58), 928);
    UNITTEST_EQ(ark::math::lcm(58, 32), 928);
    UNITTEST_EQ(ark::math::lcm(72, 78), 936);

    return ark::unittest::SUCCESS;
}

int main() {
    UNITTEST(test_utils_math);
    return 0;
}
