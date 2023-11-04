// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "tensor.h"

#include "include/ark.h"
#include "unittest/unittest_utils.h"

ark::unittest::State test_tensor_reshape_helper() {
    {
        ark::Dims shape(4, 2, 1024);
        ark::Dims ldims(4, 8, 1024);
        ark::Dims offs(1, 0, 0);
        ark::Dims new_shape(4, 1, 2048);
        ark::Dims new_ldims;
        ark::Dims new_offs;

        bool ret = ark::tensor_reshape_helper(shape, ldims, offs, new_shape,
                                              new_ldims, new_offs);
        UNITTEST_TRUE(ret);
        UNITTEST_EQ(new_ldims,
                    ark::Dims(4, 1, 8192));  // (4, 4, 2048) is ok too
        UNITTEST_EQ(new_offs, ark::Dims(1, 0, 0));
    }
    {
        ark::Dims shape(4, 64, 32);
        ark::Dims ldims(4, 128, 256);
        ark::Dims offs(0, 0, 32);
        ark::Dims new_shape(4, 64, 4, 8);
        ark::Dims new_ldims;
        ark::Dims new_offs;

        bool ret = ark::tensor_reshape_helper(shape, ldims, offs, new_shape,
                                              new_ldims, new_offs);
        UNITTEST_TRUE(ret);
        UNITTEST_EQ(new_ldims, ark::Dims(4, 128, 32, 8));
        UNITTEST_EQ(new_offs, ark::Dims(0, 0, 4, 0));
    }
    {
        ark::Dims shape(1, 1, 16384);
        ark::Dims ldims(1, 64, 16384);
        ark::Dims offs(0, 0, 0);
        ark::Dims new_shape(1, 16384);
        ark::Dims new_ldims;
        ark::Dims new_offs;

        bool ret = ark::tensor_reshape_helper(shape, ldims, offs, new_shape,
                                              new_ldims, new_offs);
        UNITTEST_TRUE(ret);
        UNITTEST_EQ(new_ldims, ark::Dims(64, 16384));
        UNITTEST_EQ(new_offs, ark::Dims(0, 0));
    }
    return ark::unittest::SUCCESS;
}

int main() {
    ark::init();
    UNITTEST(test_tensor_reshape_helper);
    return 0;
}
