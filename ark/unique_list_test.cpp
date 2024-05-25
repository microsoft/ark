// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "unique_list.hpp"

#include "unittest/unittest_utils.h"

ark::unittest::State test_unique_list() {
    ark::UniqueList<int> list;
    list.push_back(1);
    list.push_back(2);
    list.push_back(3);
    list.push_back(1);
    list.push_back(2);
    list.push_back(3);
    UNITTEST_EQ(list.size(), 3);
    UNITTEST_EQ(list[0], 1);
    UNITTEST_EQ(list[1], 2);
    UNITTEST_EQ(list[2], 3);

    list.clear();
    UNITTEST_EQ(list.size(), 0);

    list.push_back(1);
    list.push_back(2);
    list.push_back(3);
    list.push_back(1);
    list.push_back(2);
    list.push_back(3);
    list.push_back(4);
    UNITTEST_EQ(list.size(), 4);
    UNITTEST_EQ(list[0], 1);
    UNITTEST_EQ(list[1], 2);
    UNITTEST_EQ(list[2], 3);
    UNITTEST_EQ(list[3], 4);

    list.clear();
    UNITTEST_EQ(list.size(), 0);

    list.push_back(1);
    list.push_back(2);
    list.push_back(3);

    list.erase(1);
    UNITTEST_EQ(list.size(), 2);
    UNITTEST_EQ(list[0], 2);
    UNITTEST_EQ(list[1], 3);

    list.clear();
    UNITTEST_EQ(list.size(), 0);

    list.push_back(1);
    list.push_back(2);
    list.push_back(3);

    list.erase(0);
    UNITTEST_EQ(list.size(), 3);
    UNITTEST_EQ(list[0], 1);
    UNITTEST_EQ(list[1], 2);
    UNITTEST_EQ(list[2], 3);

    list.clear();
    UNITTEST_EQ(list.size(), 0);

    list.push_back(1);
    list.push_back(2);
    list.push_back(3);

    list.erase(2);
    UNITTEST_EQ(list.size(), 2);
    UNITTEST_EQ(list[0], 1);
    UNITTEST_EQ(list[1], 3);

    return ark::unittest::SUCCESS;
}

int main() {
    UNITTEST(test_unique_list);
    return 0;
}
