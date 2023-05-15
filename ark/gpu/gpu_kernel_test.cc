// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "ark/gpu/gpu_kernel.h"
#include "ark/include/ark.h"
#include "ark/process.h"
#include "ark/unittest/unittest_utils.h"

using namespace std;
using namespace ark;

//
const string test_kernel_loop_void =
    "__device__ void ark_loop_body(int _iter) {\n"
    "  // Do nothing. Print iteration counter.\n"
    "  if (threadIdx.x == 0 && blockIdx.x == 0)\n"
    "    if (_iter % 50 == 49) {\n"
    "      printf(\".\\n\");\n"
    "    } else {\n"
    "      printf(\".\");\n"
    "    }\n"
    "}\n";

//
unittest::State test_gpu_kernel_loop_void()
{
    int pid = proc_spawn([] {
        GpuMgr *mgr = get_gpu_mgr(0);
        GpuMgrCtx *ctx = mgr->create_context("test_loop_void", 0, 1);
        ctx->freeze();

        GpuLoopKernel glk{"test_kernel_loop_void",
                          {test_kernel_loop_void},
                          (unsigned int)mgr->get_gpu_info().num_sm,
                          1,
                          0,
                          "",
                          ctx,
                          1};
        glk.compile(mgr->get_gpu_info());
        glk.load();

        GpuState ret = glk.launch(ctx->create_stream());
        UNITTEST_EQ(ret, 0);
        glk.run(100);
        glk.stop();

        return 0;
    });
    UNITTEST_NE(pid, -1);
    int ret = proc_wait(pid);
    UNITTEST_EQ(ret, 0);
    return unittest::SUCCESS;
}

int main()
{
    ark::init();
    UNITTEST(test_gpu_kernel_loop_void);
    return 0;
}
