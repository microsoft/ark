// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "ark/error.hpp"

#include "unittest/unittest_utils.h"

ark::unittest::State test_error() {
    UNITTEST_THROW(throw ark::ModelError("test"), ark::ModelError);

    try {
        throw ark::ModelError("test");
    } catch (const ark::ModelError &e) {
        UNITTEST_EQ(std::string(e.what()), "test");
    }

    try {
        throw ark::ModelError("test");
    } catch (const ark::BaseError &e) {
        UNITTEST_EQ(std::string(e.what()), "test");
    }

    return ark::unittest::SUCCESS;
}

int main() {
    UNITTEST(test_error);
    return 0;
}
