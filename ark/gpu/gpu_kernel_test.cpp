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

//
// const std::string test_kernel_loop_void =
//     "__device__ void ark_loop_body(char *_buf, int _iter) {\n"
//     "  // Do nothing. Print iteration counter.\n"
//     "  if (threadIdx.x == 0 && blockIdx.x == 0) {\n"
//     "    if (_iter % 50 == 49) {\n"
//     "      printf(\".\\n\");\n"
//     "    } else {\n"
//     "      printf(\".\");\n"
//     "    }\n"
//     "  }\n"
//     "}\n";

// ark::unittest::State test_gpu_loop_kernel() {
//     int num_sm = ark::GpuManager::get_instance(0)->info().num_sm;
//     ark::GpuLoopKernel glk(0, "test_kernel_loop_void", test_kernel_loop_void,
//                            static_cast<size_t>(num_sm), 1, 0, 0);
//     glk.compile();
//     glk.load();

//     glk.launch();
//     glk.run(100);
//     glk.stop();

//     return ark::unittest::SUCCESS;
// }

int main() {
    UNITTEST(test_gpu_kernel);
    // UNITTEST(test_gpu_loop_kernel);
    return 0;
}
