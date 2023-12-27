// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "include/ark.h"

#include "file_io.h"
#include "unittest/unittest_utils.h"

ark::unittest::State test_ark_version() {
    auto version = ark::version();

    // Check if the version string is in the correct format.
    auto dot1 = version.find('.');
    auto dot2 = version.find('.', dot1 + 1);
    UNITTEST_NE(dot1, std::string::npos);
    UNITTEST_NE(dot2, std::string::npos);

    return ark::unittest::SUCCESS;
}

ark::unittest::State test_ark_init() {
    // invalid tmp directory
    ::setenv("ARK_TMP", "", 1);
    UNITTEST_THROW(ark::init(), ark::SystemError);

    // create a tmp directory
    ::setenv("ARK_TMP", "/tmp/ark/.test_ark_init", 1);
    ::setenv("ARK_KEEP_TMP", "1", 1);
    ark::init();

    // create a tmp file
    ark::write_file("/tmp/ark/.test_ark_init/test", "test");

    // clear the tmp directory
    ::setenv("ARK_KEEP_TMP", "0", 1);
    ark::init();
    UNITTEST_TRUE(!ark::is_exist("/tmp/ark/.test_ark_init/test"));

    // given tmp directory is not a directory
    ::setenv("ARK_TMP", "/dev/null", 1);
    UNITTEST_THROW(ark::init(), ark::SystemError);

    return ark::unittest::SUCCESS;
}

int main() {
    UNITTEST(test_ark_version);
    UNITTEST(test_ark_init);
    return 0;
}
