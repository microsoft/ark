// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "gpu/gpu_kernel.h"
#include "include/ark.h"
#include "include/ark_utils.h"
#include "logging.h"
#include "unittest/unittest_utils.h"

using namespace std;

//
void test_scale_internal(unsigned int bs, unsigned int n, unsigned int m,
                         float val = 0.7)
{
    ark::GpuMgr *mgr = ark::get_gpu_mgr(0);
    ark::GpuMgrCtx *ctx = mgr->create_context("test_simple_scale", 0, 1);

    unsigned int len = bs * m * n;
    ark::GpuBuf *buf_x = ctx->mem_alloc(len * sizeof(ark::half_t));
    ark::GpuBuf *buf_y = ctx->mem_alloc(len * sizeof(ark::half_t));

    ctx->freeze();

    ark::GpuKernel gk{"simple_scale",
                      {ark::unittest::get_kernel_code("simple_scale")},
                      {(unsigned int)mgr->get_gpu_info().num_sm, 1, 1},
                      {512, 1, 1},
                      0,
                      {buf_y, buf_x},
                      {},
                      {
                          {&val, sizeof(val)},
                          {&len, sizeof(len)},
                      },
                      ""};
    gk.compile(mgr->get_gpu_info());
    gk.load();

    // Set data.
    ark::srand();
    auto data_x = ark::utils::rand_halfs(len, 0.01);
    ark::gpu_memcpy(buf_x, data_x.get(), len * sizeof(ark::half_t));

    // Run the GPU kernel.
    ark::GpuStream s = ctx->create_stream();
    int ret = gk.launch(s);
    UNITTEST_EQ(ret, 0);
    ret = ctx->sync_stream(s);
    UNITTEST_EQ(ret, 0);

    // Copy the ground truth results into CPU memory.
    void *gt = malloc(len * sizeof(ark::half_t));
    UNITTEST_NE(gt, (void *)nullptr);
    ark::gpu_memcpy(gt, buf_y, len * sizeof(ark::half_t));

    mgr->destroy_context(ctx);

    //
    ark::Model model;
    ark::Tensor *tns_x = model.tensor({bs, n, m}, ark::FP16);
    ark::Tensor *tns_y = model.scale(tns_x, val);

    //
    ark::Executor exe{0, 0, 1, model, "test_scale"};
    exe.compile();

    // Set data.
    exe.tensor_memcpy(tns_x, data_x.get(), len * sizeof(ark::half_t));

    exe.launch();
    exe.run(1);
    exe.stop();

    // Copy results of the loop kernel routine into CPU memory.
    void *res = malloc(len * sizeof(ark::half_t));
    UNITTEST_NE(res, (void *)nullptr);
    exe.tensor_memcpy(res, tns_y, len * sizeof(ark::half_t));

    // Compare results with the ground truth.
    auto p =
        ark::utils::cmp_matrix((ark::half_t *)gt, (ark::half_t *)res, m, n, bs);
    float max_err = p.second;
    LOG(ark::INFO, "scale:", n, 'x', m, ",bs=", bs, setprecision(4), " mse ",
        p.first, " max_err ", max_err * 100, "%");

    free(res);
    free(gt);

    UNITTEST_EQ(max_err, 0.0);
}

ark::unittest::State test_scale()
{
    test_scale_internal(1, 1, 64);
    test_scale_internal(1, 128, 128);
    test_scale_internal(1, 4096, 1024);
    test_scale_internal(1, 1024, 4096);
    test_scale_internal(2, 1, 64);
    test_scale_internal(2, 128, 128);
    test_scale_internal(8, 4096, 1024);
    test_scale_internal(8, 1024, 4096);
    return ark::unittest::SUCCESS;
}

int main()
{
    ark::init();
    UNITTEST(test_scale);
    return ark::unittest::SUCCESS;
}
