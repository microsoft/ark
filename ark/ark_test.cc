// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "include/ark.h"

#include "unittest/unittest_utils.h"

ark::unittest::State test_version() {
    auto version = ark::version();

    // Check if the version string is in the correct format.
    auto dot1 = version.find('.');
    auto dot2 = version.find('.', dot1 + 1);
    UNITTEST_NE(dot1, std::string::npos);
    UNITTEST_NE(dot2, std::string::npos);

    return ark::unittest::SUCCESS;
}

int main() {
    ark::init();
    UNITTEST(test_version);
    return 0;
}
