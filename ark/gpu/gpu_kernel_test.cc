// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "gpu/gpu_kernel.h"

#include "gpu/gpu_loop_kernel.h"
#include "include/ark.h"
#include "unittest/unittest_utils.h"

const std::string void_kernel = "extern \"C\" __global__ void kernel() {}";

ark::unittest::State test_gpu_kernel() {
    auto ctx = ark::GpuContext::get_context(0, 1);
    ark::GpuKernel kernel(ctx, void_kernel, {1, 1, 1}, {1, 1, 1}, 0, "kernel");
    kernel.compile();
    return ark::unittest::SUCCESS;
}

//
const std::string test_kernel_loop_void =
    "__device__ void ark_loop_body(int _iter) {\n"
    "  // Do nothing. Print iteration counter.\n"
    "  if (threadIdx.x == 0 && blockIdx.x == 0) {\n"
    "    if (_iter % 50 == 49) {\n"
    "      printf(\".\\n\");\n"
    "    } else {\n"
    "      printf(\".\");\n"
    "    }\n"
    "  }\n"
    "}\n";

ark::unittest::State test_gpu_loop_kernel() {
    auto ctx = ark::GpuContext::get_context(0, 1);
    ctx->freeze();

    ark::GpuLoopKernel glk{ctx,
                           "test_kernel_loop_void",
                           {test_kernel_loop_void},
                           ctx->get_gpu_manager()->info().num_sm,
                           1,
                           0};
    glk.compile();
    glk.load();

    ark::GpuState ret = glk.launch(ctx->get_gpu_manager()->create_stream());
    UNITTEST_EQ(ret, 0);
    glk.run(100);
    glk.stop();

    return ark::unittest::SUCCESS;
}

int main() {
    ark::init();
    UNITTEST(test_gpu_kernel);
    UNITTEST(test_gpu_loop_kernel);
    return 0;
}
