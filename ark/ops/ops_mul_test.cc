// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "ark/include/ark.h"
#include "ark/unittest/unittest_utils.h"
#include "ops_test_common.h"

ark::unittest::State test_mul_fp32()
{
    test_bcast_fp32("mul", 1, 1, 2);
    test_bcast_fp32("mul", 1, 1, 64);
    test_bcast_fp32("mul", 1, 128, 128);
    test_bcast_fp32("mul", 1, 1024, 512);
    test_bcast_fp32("mul", 1, 512, 1024);
    test_bcast_fp32("mul", 2, 1, 64);
    test_bcast_fp32("mul", 2, 128, 128);
    test_bcast_fp32("mul", 4, 1024, 512);
    test_bcast_fp32("mul", 4, 512, 1024);
    return ark::unittest::SUCCESS;
}

ark::unittest::State test_mul_fp16()
{
    test_bcast_fp16("mul", 1, 1, 2);
    test_bcast_fp16("mul", 1, 1, 64);
    test_bcast_fp16("mul", 1, 128, 128);
    test_bcast_fp16("mul", 1, 1024, 512);
    test_bcast_fp16("mul", 1, 512, 1024);
    test_bcast_fp16("mul", 2, 1, 64);
    test_bcast_fp16("mul", 2, 128, 128);
    test_bcast_fp16("mul", 4, 1024, 512);
    test_bcast_fp16("mul", 4, 512, 1024);
    return ark::unittest::SUCCESS;
}

ark::unittest::State test_mul_overwrite()
{
    test_bcast_fp32("mul", 2, 1024, 512, true);
    test_bcast_fp16("mul", 2, 1024, 512, true);
    return ark::unittest::SUCCESS;
}

int main()
{
    ark::init();
    UNITTEST(test_mul_fp32);
    UNITTEST(test_mul_fp16);
    UNITTEST(test_mul_overwrite);
    return ark::unittest::SUCCESS;
}
