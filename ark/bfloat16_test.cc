// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "bfloat16.h"

#include "include/ark.h"
#include "unittest/unittest_utils.h"

ark::unittest::State test_bfloat16() {
    ark::bfloat16_t a(1.0f);
    ark::bfloat16_t b(2.0f);
    ark::bfloat16_t c = a + b;
    UNITTEST_EQ(float(c), 3.0f);

    ark::bfloat16_t d = a * b;
    UNITTEST_EQ(float(d), 2.0f);

    ark::bfloat16_t e = a / b;
    UNITTEST_EQ(float(e), 0.5f);

    ark::bfloat16_t f = a - b;
    UNITTEST_EQ(float(f), -1.0f);

    ark::bfloat16_t g = -a;
    UNITTEST_EQ(float(g), -1.0f);

    ark::bfloat16_t h = std::exp(a);
    UNITTEST_EQ(int(float(h) * 1e3), int(std::exp(1.0f) * 1e3));

    ark::bfloat16_t i = std::max(a, b);
    UNITTEST_EQ(float(i), 2.0f);

    ark::bfloat16_t j = std::min(a, b);
    UNITTEST_EQ(float(j), 1.0f);

    b--;
    UNITTEST_EQ(float(b), 1.0f);

    b++;
    UNITTEST_EQ(float(b), 2.0f);

    ++b;
    UNITTEST_EQ(float(b), 3.0f);

    --b;
    UNITTEST_EQ(float(b), 2.0f);

    ark::bfloat16_t k = ark::bfloat16_t(1.0f);
    UNITTEST_TRUE(k == a);
    UNITTEST_TRUE(k != b);
    UNITTEST_TRUE(k < b);
    UNITTEST_TRUE(k <= b);
    UNITTEST_TRUE(b > k);
    UNITTEST_TRUE(b >= k);

    bool sign = ark::bfloat16_t(-1.0f).signbit();
    UNITTEST_TRUE(sign);

    sign = ark::bfloat16_t(1.0f).signbit();
    UNITTEST_FALSE(sign);

    sign = ark::bfloat16_t(0.0f).signbit();
    UNITTEST_FALSE(sign);

    sign = ark::bfloat16_t(-0.0f).signbit();
    UNITTEST_TRUE(sign);

    ark::bfloat16_t l = std::abs(ark::bfloat16_t(-1.0f));
    UNITTEST_EQ(float(l), 1.0f);

    ark::bfloat16_t m = std::abs(ark::bfloat16_t(1.0f));
    UNITTEST_EQ(float(m), 1.0f);

    ark::bfloat16_t n = std::abs(ark::bfloat16_t(0.0f));
    UNITTEST_EQ(float(n), 0.0f);

    ark::bfloat16_t o = std::abs(ark::bfloat16_t(-0.0f));
    UNITTEST_EQ(float(o), 0.0f);

    ark::bfloat16_t p = std::abs(ark::bfloat16_t(-0.5f));
    UNITTEST_EQ(float(p), 0.5f);

    ark::bfloat16_t q = std::abs(ark::bfloat16_t(0.5f));
    UNITTEST_EQ(float(q), 0.5f);

    bool isnan = ark::isnan(ark::bfloat16_t(0.0f));
    UNITTEST_FALSE(isnan);

    isnan = ark::isnan(ark::bfloat16_t(-0.0f));
    UNITTEST_FALSE(isnan);

    isnan = ark::isnan(ark::bfloat16_t(1.0f));
    UNITTEST_FALSE(isnan);

    isnan = ark::isnan(ark::bfloat16_t(-1.0f));
    UNITTEST_FALSE(isnan);

    isnan = ark::isnan(ark::bfloat16_t(0.5f));
    UNITTEST_FALSE(isnan);

    isnan = ark::isnan(ark::bfloat16_t(-0.5f));
    UNITTEST_FALSE(isnan);

    isnan = ark::isnan(ark::bfloat16_t(1.0f / 0.0f));
    UNITTEST_FALSE(isnan);

    isnan = ark::isnan(ark::bfloat16_t(-1.0f / 0.0f));
    UNITTEST_FALSE(isnan);

    isnan = ark::isnan(ark::bfloat16_t(0.0f / 0.0f));
    UNITTEST_TRUE(isnan);

    bool isinf = ark::isinf(ark::bfloat16_t(0.0f));
    UNITTEST_FALSE(isinf);

    isinf = ark::isinf(ark::bfloat16_t(-0.0f));
    UNITTEST_FALSE(isinf);

    isinf = ark::isinf(ark::bfloat16_t(1.0f));
    UNITTEST_FALSE(isinf);

    isinf = ark::isinf(ark::bfloat16_t(-1.0f));
    UNITTEST_FALSE(isinf);

    isinf = ark::isinf(ark::bfloat16_t(0.5f));
    UNITTEST_FALSE(isinf);

    isinf = ark::isinf(ark::bfloat16_t(-0.5f));
    UNITTEST_FALSE(isinf);

    isinf = ark::isinf(ark::bfloat16_t(1.0f / 0.0f));
    UNITTEST_TRUE(isinf);

    isinf = ark::isinf(ark::bfloat16_t(-1.0f / 0.0f));
    UNITTEST_TRUE(isinf);

    isinf = ark::isinf(ark::bfloat16_t(0.0f / 0.0f));
    UNITTEST_FALSE(isinf);

    bool isfinite = ark::isfinite(ark::bfloat16_t(0.0f));
    UNITTEST_TRUE(isfinite);

    isfinite = ark::isfinite(ark::bfloat16_t(-0.0f));
    UNITTEST_TRUE(isfinite);

    isfinite = ark::isfinite(ark::bfloat16_t(1.0f));
    UNITTEST_TRUE(isfinite);

    isfinite = ark::isfinite(ark::bfloat16_t(-1.0f));
    UNITTEST_TRUE(isfinite);

    isfinite = ark::isfinite(ark::bfloat16_t(0.5f));
    UNITTEST_TRUE(isfinite);

    isfinite = ark::isfinite(ark::bfloat16_t(-0.5f));
    UNITTEST_TRUE(isfinite);

    isfinite = ark::isfinite(ark::bfloat16_t(1.0f / 0.0f));
    UNITTEST_FALSE(isfinite);

    isfinite = ark::isfinite(ark::bfloat16_t(-1.0f / 0.0f));
    UNITTEST_FALSE(isfinite);

    isfinite = ark::isfinite(ark::bfloat16_t(0.0f / 0.0f));
    UNITTEST_FALSE(isfinite);

    ark::bfloat16_t r = ark::nan_bf16("0");
    UNITTEST_TRUE(ark::isnan(r));

    bool isnormal = ark::isnormal(ark::bfloat16_t(0.0f));
    UNITTEST_FALSE(isnormal);

    isnormal = ark::isnormal(ark::bfloat16_t(-0.0f));
    UNITTEST_FALSE(isnormal);

    isnormal = ark::isnormal(ark::bfloat16_t(1.0f));
    UNITTEST_TRUE(isnormal);

    isnormal = ark::isnormal(ark::bfloat16_t(-1.0f));
    UNITTEST_TRUE(isnormal);

    isnormal = ark::isnormal(ark::bfloat16_t(0.5f));
    UNITTEST_TRUE(isnormal);

    isnormal = ark::isnormal(ark::bfloat16_t(-0.5f));
    UNITTEST_TRUE(isnormal);

    isnormal = ark::isnormal(ark::bfloat16_t(1.0f / 0.0f));
    UNITTEST_FALSE(isnormal);

    isnormal = ark::isnormal(ark::bfloat16_t(-1.0f / 0.0f));
    UNITTEST_FALSE(isnormal);

    isnormal = ark::isnormal(ark::bfloat16_t(0.0f / 0.0f));
    UNITTEST_FALSE(isnormal);

    int fpclassify = ark::fpclassify(ark::bfloat16_t(0.0f));
    UNITTEST_EQ(fpclassify, FP_ZERO);

    fpclassify = ark::fpclassify(ark::bfloat16_t(-0.0f));
    UNITTEST_EQ(fpclassify, FP_ZERO);

    fpclassify = ark::fpclassify(ark::bfloat16_t(1.0f));
    UNITTEST_EQ(fpclassify, FP_NORMAL);

    fpclassify = ark::fpclassify(ark::bfloat16_t(-1.0f));
    UNITTEST_EQ(fpclassify, FP_NORMAL);

    fpclassify = ark::fpclassify(ark::bfloat16_t(0.5f));
    UNITTEST_EQ(fpclassify, FP_NORMAL);

    fpclassify = ark::fpclassify(ark::bfloat16_t(-0.5f));
    UNITTEST_EQ(fpclassify, FP_NORMAL);

    fpclassify = ark::fpclassify(ark::bfloat16_t(1.0f / 0.0f));
    UNITTEST_EQ(fpclassify, FP_INFINITE);

    fpclassify = ark::fpclassify(ark::bfloat16_t(-1.0f / 0.0f));
    UNITTEST_EQ(fpclassify, FP_INFINITE);

    fpclassify = ark::fpclassify(ark::bfloat16_t(0.0f / 0.0f));
    UNITTEST_EQ(fpclassify, FP_NAN);

    ark::bfloat16_t s =
        ark::copysign(ark::bfloat16_t(1.0f), ark::bfloat16_t(-1.0f));
    UNITTEST_EQ(float(s), -1.0f);

    return ark::unittest::SUCCESS;
}

int main() {
    ark::init();
    UNITTEST(test_bfloat16);
    return 0;
}
