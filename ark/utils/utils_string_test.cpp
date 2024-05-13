// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "utils/utils_string.hpp"

#include "unittest/unittest_utils.h"

ark::unittest::State test_utils_string() {
    UNITTEST_TRUE(ark::is_pascal("PascalCase"));
    UNITTEST_FALSE(ark::is_pascal(""));
    UNITTEST_FALSE(ark::is_pascal("notPascalCase"));
    UNITTEST_FALSE(ark::is_pascal("Not_PascalCase"));

    UNITTEST_EQ(ark::pascal_to_snake("PascalCase"), "pascal_case");

    UNITTEST_EQ(ark::to_upper("upper"), "UPPER");
    UNITTEST_EQ(ark::to_lower("UPPER"), "upper");

    return ark::unittest::SUCCESS;
}

int main() {
    UNITTEST(test_utils_string);
    return 0;
}
