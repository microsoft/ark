// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "gpu/gpu_kernel.hpp"

#include "unittest/unittest_utils.h"

const std::string void_kernel = "extern \"C\" __global__ void kernel() {}";

ark::unittest::State test_gpu_kernel() {
    ark::GpuKernel kernel(0, void_kernel, {1, 1, 1}, {1, 1, 1}, 0);
    UNITTEST_TRUE(!kernel.is_compiled());
    kernel.compile();
    UNITTEST_TRUE(kernel.is_compiled());
    std::vector<void*> args;
    for (int i = 0; i < 10; i++) {
        kernel.launch("kernel", nullptr, args);
    }
    return ark::unittest::SUCCESS;
}

int main() {
    UNITTEST(test_gpu_kernel);
    return 0;
}
