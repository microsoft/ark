// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "arch.hpp"

#include "unittest/unittest_utils.h"

ark::unittest::State test_arch() {
    UNITTEST_TRUE(ark::Arch("CUDA") == ark::ARCH_CUDA);
    UNITTEST_TRUE(ark::Arch("CUDA", "80") == ark::ARCH_CUDA_80);
    UNITTEST_TRUE(ark::Arch("CUDA", "80", "XYZ").name() == "CUDA_80_XYZ");

    UNITTEST_TRUE(ark::ARCH_CUDA.belongs_to(ark::ARCH_ANY));
    UNITTEST_TRUE(ark::ARCH_ROCM.belongs_to(ark::ARCH_ANY));
    UNITTEST_TRUE(!ark::ARCH_CUDA.belongs_to(ark::ARCH_CUDA));
    UNITTEST_TRUE(!ark::ARCH_ROCM.belongs_to(ark::ARCH_CUDA));
    UNITTEST_TRUE(ark::ARCH_CUDA_80.belongs_to(ark::ARCH_CUDA));

    UNITTEST_TRUE(ark::ARCH_CUDA_90.later_than(ark::ARCH_CUDA_80));
    return ark::unittest::SUCCESS;
}

int main() {
    UNITTEST(test_arch);
    return 0;
}
