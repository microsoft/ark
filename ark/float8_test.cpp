// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "float8.h"

#include "unittest/unittest_utils.h"

// Tests for float_e4m3_t
ark::unittest::State test_float_e4m3_t() {
    ark::float_e4m3_t a(1.0f);
    ark::float_e4m3_t b(2.0f);
    ark::float_e4m3_t c = a + b;
    UNITTEST_EQ(float(c), 3.0f);

    ark::float_e4m3_t d = a * b;
    UNITTEST_EQ(float(d), 2.0f);

    ark::float_e4m3_t e = a / b;
    UNITTEST_EQ(float(e), 0.5f);

    ark::float_e4m3_t f = a - b;
    UNITTEST_EQ(float(f), -1.0f);

    ark::float_e4m3_t g = -a;
    UNITTEST_EQ(float(g), -1.0f);

    ark::float_e4m3_t i = std::max(a, b);
    UNITTEST_EQ(float(i), 2.0f);

    ark::float_e4m3_t j = std::min(a, b);
    UNITTEST_EQ(float(j), 1.0f);

    b--;
    UNITTEST_EQ(float(b), 1.0f);

    b++;
    UNITTEST_EQ(float(b), 2.0f);

    ++b;
    UNITTEST_EQ(float(b), 3.0f);

    --b;
    UNITTEST_EQ(float(b), 2.0f);

    b -= ark::float_e4m3_t(1.0f);
    UNITTEST_EQ(float(b), 1.0f);

    b *= ark::float_e4m3_t(4.0f);
    UNITTEST_EQ(float(b), 4.0f);

    b /= ark::float_e4m3_t(2.0f);
    UNITTEST_EQ(float(b), 2.0f);

    ark::float_e4m3_t k = ark::float_e4m3_t(1.0f);
    UNITTEST_TRUE(k == a);
    UNITTEST_TRUE(k != b);
    UNITTEST_TRUE(k < b);
    UNITTEST_TRUE(k <= b);
    UNITTEST_TRUE(b > k);
    UNITTEST_TRUE(b >= k);

    bool sign = ark::float_e4m3_t(-1.0f).signbit();
    UNITTEST_TRUE(sign);

    sign = ark::float_e4m3_t(1.0f).signbit();
    UNITTEST_FALSE(sign);

    sign = ark::float_e4m3_t(0.0f).signbit();
    UNITTEST_FALSE(sign);

    sign = ark::float_e4m3_t(-0.0f).signbit();
    UNITTEST_TRUE(sign);

    return ark::unittest::SUCCESS;

}

int main() {
    UNITTEST(test_float_e4m3_t);
    return 0;
}