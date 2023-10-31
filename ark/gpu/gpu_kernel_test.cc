// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "gpu/gpu_kernel.h"

#include "gpu/gpu_kernel_v2.h"
#include "include/ark.h"
#include "unittest/unittest_utils.h"

using namespace std;
using namespace ark;

//
const string test_kernel_loop_void =
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

//
unittest::State test_gpu_kernel_loop_void() {
    GpuMgr *mgr = get_gpu_mgr(0);
    GpuMgrCtx *ctx = mgr->create_context("test_loop_void", 0, 1);
    ctx->freeze();

    GpuLoopKernel glk{"test_kernel_loop_void",
                      {test_kernel_loop_void},
                      (unsigned int)mgr->get_gpu_info().num_sm,
                      1,
                      0,
                      "",
                      ctx};
    glk.compile(mgr->get_gpu_info());
    glk.load();

    GpuState ret = glk.launch(ctx->create_stream());
    UNITTEST_EQ(ret, 0);
    glk.run(100);
    glk.stop();

    mgr->destroy_context(ctx);

    return unittest::SUCCESS;
}

const std::string void_kernel = "__global__ void kernel() {}";

unittest::State test_gpu_kernel() {
    auto gmgr = std::make_shared<ark::GpuMgrV2>(0);
    ark::GpuKernelV2 kernel(gmgr, void_kernel, {1, 1, 1}, {1, 1, 1}, 0,
                            "kernel");
    return unittest::SUCCESS;
}

int main() {
    ark::init();
    UNITTEST(test_gpu_kernel_loop_void);
    UNITTEST(test_gpu_kernel);
    return 0;
}
