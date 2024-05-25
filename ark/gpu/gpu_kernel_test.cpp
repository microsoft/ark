// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "gpu/gpu_kernel.h"

#include "unittest/unittest_utils.h"

const std::string void_kernel = "extern \"C\" __global__ void kernel() {}";

ark::unittest::State test_gpu_kernel() {
    ark::GpuKernel kernel(0, void_kernel, {1, 1, 1}, {1, 1, 1}, 0, "kernel");
    kernel.compile();
    return ark::unittest::SUCCESS;
}

int main() {
    UNITTEST(test_gpu_kernel);
    return 0;
}
