// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

// #include <fstream>

#include "ark/gpu/gpu_kernel.h"
#include "ark/include/ark.h"
#include "ark/include/ark_utils.h"
#include "ark/logging.h"
#include "ark/unittest/unittest_utils.h"

using namespace std;

//
void test_dot_internal(unsigned int len)
{
    ark::GpuMgr *mgr = ark::get_gpu_mgr(0);
    ark::GpuMgrCtx *ctx = mgr->create_context("test_simple_dot", 0, 1);

    ark::GpuBuf *buf_a = ctx->mem_alloc(len * sizeof(half_t));
    ark::GpuBuf *buf_b = ctx->mem_alloc(len * sizeof(half_t));
    ark::GpuBuf *buf_c = ctx->mem_alloc(4);

    ctx->freeze();

    ark::GpuKernel gk{"simple_dot",
                      {get_kernel_code("simple_dot")},
                      {1, 1, 1},
                      {128, 1, 1},
                      0,
                      {buf_c, buf_a, buf_b},
                      {},
                      {
                          {&len, sizeof(len)},
                      },
                      ""};
    gk.compile(mgr->get_gpu_info());
    gk.load();

    ark::srand();
    float *res = (float *)malloc(4);
    UNITTEST_NE(res, (float *)nullptr);
    half_t *gt = (half_t *)malloc(2);
    UNITTEST_NE(gt, (half_t *)nullptr);

    for (int iter = 0; iter < 10; ++iter) {
        // Set data.
        auto data_a = rand_halfs(len, 0.1);
        auto data_b = rand_halfs(len, 0.1);

        // ifstream data("/home/changho/ark2/data_3");
        // for (unsigned int i = 0; i < len; ++i) {
        //     string line;
        //     getline(data, line);
        //     uint16_t da = (uint16_t)strtoul(line.substr(0, 4).c_str(), 0,
        //     16); uint16_t db = (uint16_t)strtoul(line.substr(5, 4).c_str(),
        //     0, 16); data_a.get()[i] = cutlass::half_t::bitcast(da);
        //     data_b.get()[i] = cutlass::half_t::bitcast(db);
        // }

        ark::gpu_memcpy(buf_a, data_a.get(), len * sizeof(half_t));
        ark::gpu_memcpy(buf_b, data_b.get(), len * sizeof(half_t));

        // Run the GPU kernel.
        ark::GpuStream s = ctx->create_stream();
        int ret = gk.launch(s);
        UNITTEST_EQ(ret, 0);
        ret = ctx->sync_stream(s);
        UNITTEST_EQ(ret, 0);

        // Copy the result into CPU memory.
        ark::gpu_memcpy(res, buf_c, 4);

        // Calculate the ground truth.
        *gt = half_t(0);
        // cout << *gt << endl;
        for (unsigned int i = 0; i < len; ++i) {
            *gt += data_a.get()[i] * data_b.get()[i];
            // cout << *gt << endl;
        }

        float err = error_rate(half_t(*gt), half_t(*res));

        LOG(ark::INFO, "dot:", len, setprecision(4), " res ", *res, " gt ", *gt,
            " err ", err * 100, "%");

        UNITTEST_TRUE(err < 0.01);

        // if (((*res) * (*gt) < 0) &&
        //     (abs(*res - *gt) > (float)numeric_limits<half_t>::epsilon())) {
        //     cout << hex;
        //     for (unsigned int i = 0; i < len; ++i) {
        //         cout << data_a.get()[i].storage << "," <<
        //         data_b.get()[i].storage << endl;
        //     }
        //     cout << dec;
        //     UNITTEST_TRUE(false);
        // }
    }

    free(res);
    free(gt);
    mgr->destroy_context(ctx);
}

ark::unittest::State test_dot()
{
    test_dot_internal(1024);
    return ark::unittest::SUCCESS;
}

int main()
{
    ark::init();
    UNITTEST(test_dot);
    return ark::unittest::SUCCESS;
}
