// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "range.hpp"

#include "unittest/unittest_utils.h"

ark::unittest::State test_range() {
    ark::Range r1(0, 10);
    UNITTEST_EQ(r1.step(), 1);
    int v = 0;
    int cnt = 0;
    for (auto i : r1) {
        UNITTEST_EQ(i, v++);
        cnt++;
    }
    UNITTEST_EQ(cnt, 10);

    ark::Range r2(0, 10, 3);
    UNITTEST_EQ(r2.step(), 3);
    v = 0;
    cnt = 0;
    for (auto i : r2) {
        UNITTEST_EQ(i, v);
        v += 3;
        cnt++;
    }
    UNITTEST_EQ(cnt, 4);

    ark::Range r3(13, 1, -3);
    UNITTEST_EQ(r3.step(), -3);
    v = 13;
    cnt = 0;
    for (auto i : r3) {
        UNITTEST_EQ(i, v);
        v -= 3;
        cnt++;
    }
    UNITTEST_EQ(cnt, 4);

    ark::Range r4(0, 0);
    UNITTEST_EQ(r4.step(), 1);
    cnt = 0;
    for ([[maybe_unused]] auto i : r4) {
        cnt++;
    }
    UNITTEST_EQ(cnt, 0);

    ark::Range r5(2, 19, 3);  // 2, 5, 8, 11, 14, 17
    ark::Range r6(3, 17, 2);  // 3, 5, 7, 9, 11, 13, 15
    auto intersec = r5.intersection(r6);
    UNITTEST_EQ(intersec[0], 5);
    UNITTEST_EQ(intersec[1], 11);

    return ark::unittest::State::SUCCESS;
}

int main() {
    UNITTEST(test_range);
    return 0;
}
