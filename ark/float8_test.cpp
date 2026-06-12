// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "float8.h"

#include <cmath>
#include <limits>

#include "unittest/unittest_utils.h"

// Tests for float_e4m3_t
ark::unittest::State test_float_e4m3_t() {
    // Basic arithmetic
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

    // Post/pre increment/decrement
    b--;
    UNITTEST_EQ(float(b), 1.0f);

    b++;
    UNITTEST_EQ(float(b), 2.0f);

    ++b;
    UNITTEST_EQ(float(b), 3.0f);

    --b;
    UNITTEST_EQ(float(b), 2.0f);

    // Compound assignment
    b -= ark::float_e4m3_t(1.0f);
    UNITTEST_EQ(float(b), 1.0f);

    b *= ark::float_e4m3_t(4.0f);
    UNITTEST_EQ(float(b), 4.0f);

    b /= ark::float_e4m3_t(2.0f);
    UNITTEST_EQ(float(b), 2.0f);

    b += ark::float_e4m3_t(1.0f);
    UNITTEST_EQ(float(b), 3.0f);

    // Comparison operators
    ark::float_e4m3_t k = ark::float_e4m3_t(1.0f);
    b = ark::float_e4m3_t(2.0f);
    UNITTEST_TRUE(k == a);
    UNITTEST_TRUE(k != b);
    UNITTEST_TRUE(k < b);
    UNITTEST_TRUE(k <= b);
    UNITTEST_TRUE(b > k);
    UNITTEST_TRUE(b >= k);
    UNITTEST_TRUE(k <= a);
    UNITTEST_TRUE(k >= a);

    // Sign bit
    bool sign = ark::float_e4m3_t(-1.0f).signbit();
    UNITTEST_TRUE(sign);

    sign = ark::float_e4m3_t(1.0f).signbit();
    UNITTEST_FALSE(sign);

    sign = ark::float_e4m3_t(0.0f).signbit();
    UNITTEST_FALSE(sign);

    sign = ark::float_e4m3_t(-0.0f).signbit();
    UNITTEST_TRUE(sign);

    // Round-trip: normal values
    UNITTEST_EQ(float(ark::float_e4m3_t(0.5f)), 0.5f);
    UNITTEST_EQ(float(ark::float_e4m3_t(1.0f)), 1.0f);
    UNITTEST_EQ(float(ark::float_e4m3_t(2.0f)), 2.0f);
    UNITTEST_EQ(float(ark::float_e4m3_t(4.0f)), 4.0f);
    UNITTEST_EQ(float(ark::float_e4m3_t(-1.0f)), -1.0f);
    UNITTEST_EQ(float(ark::float_e4m3_t(-2.0f)), -2.0f);

    // Round-trip: zero
    UNITTEST_EQ(float(ark::float_e4m3_t(0.0f)), 0.0f);
    UNITTEST_EQ(float(ark::float_e4m3_t(-0.0f)), -0.0f);

    // Round-trip: subnormals
    // E4M3 smallest subnormal: 2^(-6) * 2^(-3) = 2^(-9) = 0.001953125
    ark::float_e4m3_t sub = ark::float_e4m3_t::bitcast(0x01);
    float sub_f = float(sub);
    UNITTEST_TRUE(sub_f > 0.0f);
    UNITTEST_TRUE(sub_f < 0.01f);
    // Round-trip subnormal
    ark::float_e4m3_t sub2(sub_f);
    UNITTEST_EQ(float(sub2), sub_f);

    // NaN
    ark::float_e4m3_t nan_val = ark::float_e4m3_t::bitcast(0x7f);
    UNITTEST_TRUE(std::isnan(float(nan_val)));
    // NaN from float NaN
    ark::float_e4m3_t nan_from_float(std::numeric_limits<float>::quiet_NaN());
    UNITTEST_TRUE(std::isnan(float(nan_from_float)));

    // Overflow saturation: values beyond 448.0 should clamp
    ark::float_e4m3_t sat = ark::float_e4m3_t(500.0f);
    UNITTEST_EQ(float(sat), 448.0f);
    ark::float_e4m3_t sat_neg = ark::float_e4m3_t(-500.0f);
    UNITTEST_EQ(float(sat_neg), -448.0f);

    // Max representable value
    ark::float_e4m3_t max_val = ark::float_e4m3_t::bitcast(0x7e);
    UNITTEST_EQ(float(max_val), 448.0f);

    // Inf -> saturates to max
    ark::float_e4m3_t from_inf(std::numeric_limits<float>::infinity());
    UNITTEST_EQ(float(from_inf), 448.0f);
    ark::float_e4m3_t from_neg_inf(-std::numeric_limits<float>::infinity());
    UNITTEST_EQ(float(from_neg_inf), -448.0f);

    // Type casts
    UNITTEST_EQ(double(ark::float_e4m3_t(3.0f)), 3.0);
    UNITTEST_EQ(int(ark::float_e4m3_t(3.0f)), 3);
    UNITTEST_TRUE(bool(ark::float_e4m3_t(1.0f)));
    UNITTEST_TRUE(bool(ark::float_e4m3_t(0.5f)));
    UNITTEST_FALSE(bool(ark::float_e4m3_t(0.0f)));

    // Cross-type conversion E5M2 -> E4M3
    ark::float_e5m2_t e5(2.0f);
    ark::float_e4m3_t from_e5(e5);
    UNITTEST_EQ(float(from_e5), 2.0f);

    // User-defined literal
    auto lit = 3.0_fe4m3;
    UNITTEST_EQ(float(lit), 3.0f);
    auto lit_int = 4_fe4m3;
    UNITTEST_EQ(float(lit_int), 4.0f);

    return ark::unittest::SUCCESS;
}

// Tests for numeric_limits<float_e4m3_t>
ark::unittest::State test_float_e4m3_limits() {
    using lim = std::numeric_limits<ark::float_e4m3_t>;
    UNITTEST_TRUE(lim::is_specialized);
    UNITTEST_TRUE(lim::is_signed);
    UNITTEST_FALSE(lim::is_integer);
    UNITTEST_FALSE(lim::is_exact);

    // E4M3 has no infinity (OCP spec)
    UNITTEST_FALSE(lim::has_infinity);
    UNITTEST_TRUE(lim::has_quiet_NaN);

    // lowest() must be negative: −448.0
    float lowest = float(lim::lowest());
    UNITTEST_TRUE(lowest < 0.0f);
    UNITTEST_EQ(lowest, -448.0f);

    // max() = 448.0
    float max_val = float(lim::max());
    UNITTEST_EQ(max_val, 448.0f);

    // min() > 0 (smallest positive normal)
    float min_val = float(lim::min());
    UNITTEST_TRUE(min_val > 0.0f);

    // quiet_NaN is NaN
    UNITTEST_TRUE(std::isnan(float(lim::quiet_NaN())));

    // denorm_min > 0
    UNITTEST_TRUE(float(lim::denorm_min()) > 0.0f);

    // epsilon > 0
    UNITTEST_TRUE(float(lim::epsilon()) > 0.0f);

    return ark::unittest::SUCCESS;
}

// Tests for float_e5m2_t
ark::unittest::State test_float_e5m2_t() {
    // Basic arithmetic
    ark::float_e5m2_t a(1.0f);
    ark::float_e5m2_t b(2.0f);
    ark::float_e5m2_t c = a + b;
    UNITTEST_EQ(float(c), 3.0f);

    ark::float_e5m2_t d = a * b;
    UNITTEST_EQ(float(d), 2.0f);

    ark::float_e5m2_t e = a / b;
    UNITTEST_EQ(float(e), 0.5f);

    ark::float_e5m2_t f = a - b;
    UNITTEST_EQ(float(f), -1.0f);

    ark::float_e5m2_t g = -a;
    UNITTEST_EQ(float(g), -1.0f);

    ark::float_e5m2_t i = std::max(a, b);
    UNITTEST_EQ(float(i), 2.0f);

    ark::float_e5m2_t j = std::min(a, b);
    UNITTEST_EQ(float(j), 1.0f);

    // Post/pre increment/decrement
    b--;
    UNITTEST_EQ(float(b), 1.0f);

    b++;
    UNITTEST_EQ(float(b), 2.0f);

    ++b;
    UNITTEST_EQ(float(b), 3.0f);

    --b;
    UNITTEST_EQ(float(b), 2.0f);

    // Compound assignment
    b -= ark::float_e5m2_t(1.0f);
    UNITTEST_EQ(float(b), 1.0f);

    b *= ark::float_e5m2_t(4.0f);
    UNITTEST_EQ(float(b), 4.0f);

    b /= ark::float_e5m2_t(2.0f);
    UNITTEST_EQ(float(b), 2.0f);

    b += ark::float_e5m2_t(1.0f);
    // E5M2 has only 2 mantissa bits, so 3.0 = 1.10 * 2^1 = exact
    UNITTEST_EQ(float(b), 3.0f);

    // Comparison operators
    ark::float_e5m2_t k(1.0f);
    b = ark::float_e5m2_t(2.0f);
    UNITTEST_TRUE(k == a);
    UNITTEST_TRUE(k != b);
    UNITTEST_TRUE(k < b);
    UNITTEST_TRUE(k <= b);
    UNITTEST_TRUE(b > k);
    UNITTEST_TRUE(b >= k);
    UNITTEST_TRUE(k <= a);
    UNITTEST_TRUE(k >= a);

    // Sign bit
    bool sign = ark::float_e5m2_t(-1.0f).signbit();
    UNITTEST_TRUE(sign);

    sign = ark::float_e5m2_t(1.0f).signbit();
    UNITTEST_FALSE(sign);

    sign = ark::float_e5m2_t(0.0f).signbit();
    UNITTEST_FALSE(sign);

    sign = ark::float_e5m2_t(-0.0f).signbit();
    UNITTEST_TRUE(sign);

    // Round-trip: normal values
    UNITTEST_EQ(float(ark::float_e5m2_t(0.5f)), 0.5f);
    UNITTEST_EQ(float(ark::float_e5m2_t(1.0f)), 1.0f);
    UNITTEST_EQ(float(ark::float_e5m2_t(2.0f)), 2.0f);
    UNITTEST_EQ(float(ark::float_e5m2_t(4.0f)), 4.0f);
    UNITTEST_EQ(float(ark::float_e5m2_t(-1.0f)), -1.0f);
    UNITTEST_EQ(float(ark::float_e5m2_t(-2.0f)), -2.0f);

    // Round-trip: zero
    UNITTEST_EQ(float(ark::float_e5m2_t(0.0f)), 0.0f);
    UNITTEST_EQ(float(ark::float_e5m2_t(-0.0f)), -0.0f);

    // Round-trip: subnormals
    ark::float_e5m2_t sub = ark::float_e5m2_t::bitcast(0x01);
    float sub_f = float(sub);
    UNITTEST_TRUE(sub_f > 0.0f);
    // Round-trip subnormal
    ark::float_e5m2_t sub2(sub_f);
    UNITTEST_EQ(float(sub2), sub_f);

    // NaN
    ark::float_e5m2_t nan_val = ark::float_e5m2_t::bitcast(0x7f);
    UNITTEST_TRUE(std::isnan(float(nan_val)));
    // NaN from float NaN
    ark::float_e5m2_t nan_from_float(std::numeric_limits<float>::quiet_NaN());
    UNITTEST_TRUE(std::isnan(float(nan_from_float)));

    // Infinity
    ark::float_e5m2_t inf_val = ark::float_e5m2_t::bitcast(0x7c);
    UNITTEST_TRUE(std::isinf(float(inf_val)));
    UNITTEST_TRUE(float(inf_val) > 0.0f);
    // Negative infinity
    ark::float_e5m2_t neg_inf_val = ark::float_e5m2_t::bitcast(0xfc);
    UNITTEST_TRUE(std::isinf(float(neg_inf_val)));
    UNITTEST_TRUE(float(neg_inf_val) < 0.0f);
    // Inf round-trip from float
    ark::float_e5m2_t inf_from_float(std::numeric_limits<float>::infinity());
    UNITTEST_TRUE(std::isinf(float(inf_from_float)));

    // NaN propagation through arithmetic
    ark::float_e5m2_t nan_e5 = ark::float_e5m2_t::bitcast(0x7f);
    ark::float_e5m2_t one_e5(1.0f);
    UNITTEST_TRUE(std::isnan(float(nan_e5 + one_e5)));
    UNITTEST_TRUE(std::isnan(float(nan_e5 * one_e5)));
    UNITTEST_TRUE(std::isnan(float(nan_e5 - one_e5)));
    UNITTEST_TRUE(std::isnan(float(nan_e5 / one_e5)));

    // Type casts
    UNITTEST_EQ(double(ark::float_e5m2_t(3.0f)), 3.0);
    UNITTEST_EQ(int(ark::float_e5m2_t(3.0f)), 3);
    UNITTEST_TRUE(bool(ark::float_e5m2_t(1.0f)));
    UNITTEST_TRUE(bool(ark::float_e5m2_t(0.5f)));
    UNITTEST_FALSE(bool(ark::float_e5m2_t(0.0f)));

    // Cross-type conversion E4M3 -> E5M2
    ark::float_e4m3_t e4(2.0f);
    ark::float_e5m2_t from_e4(e4);
    UNITTEST_EQ(float(from_e4), 2.0f);

    // User-defined literal
    auto lit = 3.0_fe5m2;
    UNITTEST_EQ(float(lit), 3.0f);
    auto lit_int = 4_fe5m2;
    UNITTEST_EQ(float(lit_int), 4.0f);

    return ark::unittest::SUCCESS;
}

// Tests for numeric_limits<float_e5m2_t>
ark::unittest::State test_float_e5m2_limits() {
    using lim = std::numeric_limits<ark::float_e5m2_t>;
    UNITTEST_TRUE(lim::is_specialized);
    UNITTEST_TRUE(lim::is_signed);
    UNITTEST_FALSE(lim::is_integer);
    UNITTEST_FALSE(lim::is_exact);

    // E5M2 has infinity
    UNITTEST_TRUE(lim::has_infinity);
    UNITTEST_TRUE(lim::has_quiet_NaN);

    // lowest() must be negative
    float lowest = float(lim::lowest());
    UNITTEST_TRUE(lowest < 0.0f);

    // max() is positive
    float max_val = float(lim::max());
    UNITTEST_TRUE(max_val > 0.0f);

    // lowest() == -max()
    UNITTEST_EQ(lowest, -max_val);

    // min() > 0 (smallest positive normal)
    float min_val = float(lim::min());
    UNITTEST_TRUE(min_val > 0.0f);

    // infinity round-trips
    UNITTEST_TRUE(std::isinf(float(lim::infinity())));
    UNITTEST_TRUE(float(lim::infinity()) > 0.0f);

    // quiet_NaN is NaN
    UNITTEST_TRUE(std::isnan(float(lim::quiet_NaN())));

    // denorm_min > 0
    UNITTEST_TRUE(float(lim::denorm_min()) > 0.0f);

    // epsilon > 0
    UNITTEST_TRUE(float(lim::epsilon()) > 0.0f);

    return ark::unittest::SUCCESS;
}

int main() {
    UNITTEST(test_float_e4m3_t);
    UNITTEST(test_float_e4m3_limits);
    UNITTEST(test_float_e5m2_t);
    UNITTEST(test_float_e5m2_limits);
    return 0;
}
