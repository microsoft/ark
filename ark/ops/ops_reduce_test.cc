// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "ark/gpu/gpu_kernel.h"
#include "ark/include/ark.h"
#include "ark/logging.h"
#include "ark/ops/ops_test_utils.h"
#include "ark/unittest/unittest_utils.h"

using namespace std;

//
void test_reduce_internal(unsigned int n, unsigned int m, unsigned int k,
                          bool is_relu = false)
{
    ark::GpuMgr *mgr = ark::get_gpu_mgr(0);
    ark::GpuMgrCtx *ctx = mgr->create_context("test_simple_reduce", 0, 1);

    size_t buf_x_sz = (size_t)m * (size_t)n * (size_t)k * sizeof(half_t);
    size_t buf_y_sz = (size_t)m * (size_t)n * sizeof(half_t);

    ark::GpuBuf *buf_x = ctx->mem_alloc(buf_x_sz);
    ark::GpuBuf *buf_y = ctx->mem_alloc(buf_y_sz);

    ctx->freeze();

    ark::GpuKernel gk{"simple_reduce",
                      {get_kernel_code("simple_reduce")},
                      {(unsigned int)mgr->get_gpu_info().num_sm, 1, 1},
                      {128, 1, 1},
                      0,
                      {buf_y, buf_x},
                      {},
                      {{&m, sizeof(m)},
                       {&n, sizeof(n)},
                       {&k, sizeof(k)},
                       {&is_relu, sizeof(is_relu)}},
                      ""};
    gk.compile(mgr->get_gpu_info());
    gk.load();

    // Set data.
    ark::srand();
    auto data_a = rand_halfs(buf_x_sz / sizeof(half_t), 0.01);
    auto data_b = rand_halfs(buf_y_sz / sizeof(half_t), 0.01);
    ark::gpu_memcpy(buf_x, data_a.get(), buf_x_sz);

    // Run the GPU kernel.
    ark::GpuStream s = ctx->create_stream();
    int ret = gk.launch(s);
    UNITTEST_EQ(ret, 0);
    ret = ctx->sync_stream(s);
    UNITTEST_EQ(ret, 0);

    // Copy the ground truth results into CPU memory.
    void *gt = malloc(buf_y_sz);
    UNITTEST_NE(gt, (void *)nullptr);
    ark::gpu_memcpy(gt, buf_y, buf_y_sz);

    mgr->destroy_context(ctx);

    //
    ark::Model model;
    ark::Tensor *tns_x = model.tensor({k, n, m}, ark::FP16);
    ark::Tensor *tns_y = model.tensor({1, n, m}, ark::FP16);
    model.reduce(tns_x, 0, tns_y, is_relu);

    //
    ark::Executor exe{0, 0, 1, model, "test_reduce"};
    exe.compile();

    // Set data.
    exe.tensor_memcpy(tns_x, data_a.get(), buf_x_sz);

    exe.launch();
    exe.run(1);
    exe.stop();

    // Copy results of the loop kernel routine into CPU memory.
    void *res = malloc(buf_y_sz);
    UNITTEST_NE(res, (void *)nullptr);
    exe.tensor_memcpy(res, tns_y, buf_y_sz);

    // Compare results with the ground truth.
    auto p = cmp_matrix((half_t *)gt, (half_t *)res, m, n);
    float max_err = p.second;
    LOG(ark::INFO, "reduce:", n, 'x', m, 'x', k, "(relu=", is_relu, ") ",
        setprecision(4), " mse ", p.first, " max_err ", max_err * 100, "%");

    free(res);
    free(gt);

    UNITTEST_EQ(max_err, 0.0);
}

ark::unittest::State test_reduce()
{
    test_reduce_internal(1, 64, 2);
    test_reduce_internal(1, 64, 8);
    test_reduce_internal(1, 64, 9);

    test_reduce_internal(2, 64, 4);
    test_reduce_internal(8, 64, 4);
    test_reduce_internal(64, 64, 4);

    test_reduce_internal(1024, 384, 4);
    return ark::unittest::SUCCESS;
}

ark::unittest::State test_reduce_relu()
{
    test_reduce_internal(1024, 384, 4, true);
    return ark::unittest::SUCCESS;
}

int main()
{
    ark::init();
    UNITTEST(test_reduce);
    UNITTEST(test_reduce_relu);
    return ark::unittest::SUCCESS;
}
