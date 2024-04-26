// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "ark/init.hpp"

#include "file_io.h"
#include "unittest/unittest_utils.h"

ark::unittest::State test_init() {
    // invalid tmp directory
    ::setenv("ARK_TMP", "", 1);
    UNITTEST_THROW(ark::init(), ark::SystemError);

    // create a tmp directory
    ::setenv("ARK_TMP", "/tmp/ark/.test_init", 1);
    ::setenv("ARK_KEEP_TMP", "1", 1);
    ark::init();

    // create a tmp file
    ark::write_file("/tmp/ark/.test_init/test", "test");

    // clear the tmp directory
    ::setenv("ARK_KEEP_TMP", "0", 1);
    ark::init();
    UNITTEST_TRUE(!ark::is_exist("/tmp/ark/.test_init/test"));

    // given tmp directory is not a directory
    ::setenv("ARK_TMP", "/dev/null", 1);
    UNITTEST_THROW(ark::init(), ark::SystemError);

    return ark::unittest::SUCCESS;
}

int main() {
    UNITTEST(test_init);
    return 0;
}
