// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "gpu/gpu_kernel.h"
#include "include/ark.h"
#include "include/ark_utils.h"
#include "logging.h"
#include "unittest/unittest_utils.h"
#include <cassert>

using namespace std;

// m,n,k: Problem size. CAUTION: `m` and `n` are assumed to be multiple of 16.
// bs_a: Batch size of left-side matrix.
// bs_b: Batch size of right-side matrix.
// iter: Number of iterations.
void test_matmul_internal(unsigned int m, unsigned int n, unsigned int k,
                          unsigned int bs_a, unsigned int bs_b, int split_k = 1,
                          int gran_lev = -1, unsigned int iter = 1)
{
    assert(bs_a == bs_b || bs_a == 1 || bs_b == 1);
    unsigned int bs_res = bs_a > bs_b ? bs_a : bs_b;

    ark::GpuMgr *mgr = ark::get_gpu_mgr(0);
    ark::GpuMgrCtx *ctx = mgr->create_context("test_simple_matmul_nt", 0, 1);

    size_t buf_a_sz =
        (size_t)bs_a * (size_t)m * (size_t)k * sizeof(ark::half_t);
    size_t buf_b_sz =
        (size_t)bs_b * (size_t)k * (size_t)n * sizeof(ark::half_t);
    size_t buf_res_sz =
        (size_t)bs_res * (size_t)m * (size_t)n * sizeof(ark::half_t);

    // Reserved GPU buffers for execution of a manually written kernel,
    // `simple_matmul_nt`.
    ark::GpuBuf *buf_a = ctx->mem_alloc(buf_a_sz);
    ark::GpuBuf *buf_b = ctx->mem_alloc(buf_b_sz);
    ark::GpuBuf *buf_gt = ctx->mem_alloc(buf_res_sz);
    // ark::GpuBuf *buf_res = ctx->mem_alloc(buf_res_sz);

    ctx->freeze();

    bool is_relu = false;

    // Define `simple_matmul_nt` kernel to generate the ground truth.
    ark::GpuKernel gk{"simple_matmul_nt",
                      {ark::unittest::get_kernel_code("simple_matmul_nt")},
                      {n / 16, m / 16, 1},
                      {16, 16, 1},
                      0,
                      {buf_gt, buf_a, buf_b},
                      {},
                      {{&m, sizeof(m)},
                       {&n, sizeof(n)},
                       {&k, sizeof(k)},
                       {&is_relu, sizeof(is_relu)}},
                      ""};
    gk.compile(mgr->get_gpu_info());
    gk.load();

    // Generate random data for tests.
    ark::srand();
    auto data_a = ark::utils::rand_halfs(buf_a_sz / sizeof(ark::half_t), 0.001);
    auto data_b = ark::utils::rand_halfs(buf_b_sz / sizeof(ark::half_t), 0.001);
    ark::gpu_memcpy(buf_a, data_a.get(), buf_a_sz);
    ark::gpu_memcpy(buf_b, data_b.get(), buf_b_sz);

    // Run the GPU kernel.
    ark::GpuStream s = ctx->create_stream();
    int ret = gk.launch(s);
    UNITTEST_EQ(ret, 0);
    ret = ctx->sync_stream(s);
    UNITTEST_EQ(ret, 0);

    // Copy the ground truth results into CPU memory.
    void *gt = malloc(buf_res_sz);
    UNITTEST_NE(gt, (void *)nullptr);
    ark::gpu_memcpy(gt, buf_gt, buf_res_sz);

    // Declare an equivalent matmul using Model APIs.
    ark::Model model;
    ark::Tensor *tns_a = model.tensor({m, k}, ark::FP16);
    ark::Tensor *tns_b = model.tensor({k, n}, ark::FP16);
    ark::Tensor *tns_res = model.matmul(tns_a, tns_b, nullptr, split_k, false,
                                        false, "matmul", gran_lev);

    mgr->destroy_context(ctx);

    //
    ark::Executor exe{0, 0, 1, model, "test_matmul_nt"};
    exe.compile();

    tns_a->write(data_a.get());
    tns_b->write(data_b.get());

    exe.launch();
    exe.run(iter);
    float elapsed = exe.stop();

    // Copy results of the loop kernel routine into CPU memory.
    void *res = malloc(buf_res_sz);
    UNITTEST_NE(res, (void *)nullptr);
    tns_res->read(res);

    // Calculate CPU results
    // float temp;
    // unsigned int h;
    // unsigned int w;
    // for (unsigned int i = 0; i < (size_t)bs_res * (size_t)m * (size_t)n; ++i)
    // {
    //     temp = 0;
    //     h = i % m;
    //     w = i / m;
    //     for (unsigned int j = 0; j < k; ++j) {
    //         temp += (float)(data_a.get()[j * m + h]) * (float)(data_b.get()[j
    //         * n + w]);
    //     }
    //     ((ark::half_t *)gt)[i] = ark::half_t(temp);
    // }

    // Compare results with the ground truth.
    auto p =
        ark::utils::cmp_matrix((ark::half_t *)gt, (ark::half_t *)res, m, n);
    float max_err = p.second;
    LOG(ark::INFO, "matmul:", m, 'x', n, 'x', k, "(split_k=", split_k,
        ",gran_lev=", gran_lev, ") ", setprecision(4), " mse ", p.first,
        " max_err ", max_err * 100, "%", " elapsed ", elapsed, "ms iter ",
        iter);

    free(res);
    free(gt);

    UNITTEST_EQ(max_err, 0.0);
}

ark::unittest::State test_matmul_gran0()
{
    test_matmul_internal(/*m=*/64, /*n=*/64, /*k=*/32, /*bs_a=*/1, /*bs_b=*/1,
                         /*split_k=*/1, /*gran_lev=*/0);
    test_matmul_internal(/*m=*/128, /*n=*/64, /*k=*/32, /*bs_a=*/1, /*bs_b=*/1,
                         /*split_k=*/1, /*gran_lev=*/0);
    test_matmul_internal(/*m=*/64, /*n=*/128, /*k=*/32, /*bs_a=*/1, /*bs_b=*/1,
                         /*split_k=*/1, /*gran_lev=*/0);
    test_matmul_internal(/*m=*/128, /*n=*/128, /*k=*/32, /*bs_a=*/1, /*bs_b=*/1,
                         /*split_k=*/1, /*gran_lev=*/0);

    test_matmul_internal(/*m=*/64, /*n=*/64, /*k=*/64, /*bs_a=*/1, /*bs_b=*/1,
                         /*split_k=*/1, /*gran_lev=*/0);
    test_matmul_internal(/*m=*/128, /*n=*/64, /*k=*/64, /*bs_a=*/1, /*bs_b=*/1,
                         /*split_k=*/1, /*gran_lev=*/0);
    test_matmul_internal(/*m=*/64, /*n=*/128, /*k=*/64, /*bs_a=*/1, /*bs_b=*/1,
                         /*split_k=*/1, /*gran_lev=*/0);
    test_matmul_internal(/*m=*/128, /*n=*/128, /*k=*/64, /*bs_a=*/1, /*bs_b=*/1,
                         /*split_k=*/1, /*gran_lev=*/0);
    test_matmul_internal(/*m=*/256, /*n=*/128, /*k=*/64, /*bs_a=*/1, /*bs_b=*/1,
                         /*split_k=*/1, /*gran_lev=*/0);

    test_matmul_internal(/*m=*/128, /*n=*/128, /*k=*/256, /*bs_a=*/1,
                         /*bs_b=*/1, /*split_k=*/1, /*gran_lev=*/0);

    test_matmul_internal(/*m=*/128, /*n=*/4096, /*k=*/1024, /*bs_a=*/1,
                         /*bs_b=*/1, /*split_k=*/1, /*gran_lev=*/0);
    return ark::unittest::SUCCESS;
}

ark::unittest::State test_matmul_gran1()
{
    test_matmul_internal(/*m=*/64, /*n=*/64, /*k=*/32, /*bs_a=*/1, /*bs_b=*/1,
                         /*split_k=*/1, /*gran_lev=*/1);
    test_matmul_internal(/*m=*/128, /*n=*/64, /*k=*/32, /*bs_a=*/1, /*bs_b=*/1,
                         /*split_k=*/1, /*gran_lev=*/1);
    test_matmul_internal(/*m=*/64, /*n=*/128, /*k=*/32, /*bs_a=*/1, /*bs_b=*/1,
                         /*split_k=*/1, /*gran_lev=*/1);
    test_matmul_internal(/*m=*/128, /*n=*/128, /*k=*/32, /*bs_a=*/1, /*bs_b=*/1,
                         /*split_k=*/1, /*gran_lev=*/1);

    test_matmul_internal(/*m=*/64, /*n=*/64, /*k=*/64, /*bs_a=*/1, /*bs_b=*/1,
                         /*split_k=*/1, /*gran_lev=*/1);
    test_matmul_internal(/*m=*/128, /*n=*/64, /*k=*/64, /*bs_a=*/1, /*bs_b=*/1,
                         /*split_k=*/1, /*gran_lev=*/1);
    test_matmul_internal(/*m=*/64, /*n=*/128, /*k=*/64, /*bs_a=*/1, /*bs_b=*/1,
                         /*split_k=*/1, /*gran_lev=*/1);
    test_matmul_internal(/*m=*/128, /*n=*/128, /*k=*/64, /*bs_a=*/1, /*bs_b=*/1,
                         /*split_k=*/1, /*gran_lev=*/1);
    test_matmul_internal(/*m=*/256, /*n=*/128, /*k=*/64, /*bs_a=*/1, /*bs_b=*/1,
                         /*split_k=*/1, /*gran_lev=*/1);

    test_matmul_internal(/*m=*/128, /*n=*/128, /*k=*/256, /*bs_a=*/1,
                         /*bs_b=*/1, /*split_k=*/1, /*gran_lev=*/1);

    test_matmul_internal(/*m=*/128, /*n=*/4096, /*k=*/1024, /*bs_a=*/1,
                         /*bs_b=*/1, /*split_k=*/1, /*gran_lev=*/1);
    return ark::unittest::SUCCESS;
}

ark::unittest::State test_matmul_gran2()
{
    test_matmul_internal(/*m=*/64, /*n=*/64, /*k=*/32, /*bs_a=*/1, /*bs_b=*/1,
                         /*split_k=*/1, /*gran_lev=*/2);
    test_matmul_internal(/*m=*/128, /*n=*/64, /*k=*/32, /*bs_a=*/1, /*bs_b=*/1,
                         /*split_k=*/1, /*gran_lev=*/2);
    test_matmul_internal(/*m=*/64, /*n=*/128, /*k=*/32, /*bs_a=*/1, /*bs_b=*/1,
                         /*split_k=*/1, /*gran_lev=*/2);
    test_matmul_internal(/*m=*/128, /*n=*/128, /*k=*/32, /*bs_a=*/1, /*bs_b=*/1,
                         /*split_k=*/1, /*gran_lev=*/2);

    test_matmul_internal(/*m=*/64, /*n=*/64, /*k=*/64, /*bs_a=*/1, /*bs_b=*/1,
                         /*split_k=*/1, /*gran_lev=*/2);
    test_matmul_internal(/*m=*/128, /*n=*/64, /*k=*/64, /*bs_a=*/1, /*bs_b=*/1,
                         /*split_k=*/1, /*gran_lev=*/2);
    test_matmul_internal(/*m=*/64, /*n=*/128, /*k=*/64, /*bs_a=*/1, /*bs_b=*/1,
                         /*split_k=*/1, /*gran_lev=*/2);
    test_matmul_internal(/*m=*/128, /*n=*/128, /*k=*/64, /*bs_a=*/1, /*bs_b=*/1,
                         /*split_k=*/1, /*gran_lev=*/2);
    test_matmul_internal(/*m=*/256, /*n=*/128, /*k=*/64, /*bs_a=*/1, /*bs_b=*/1,
                         /*split_k=*/1, /*gran_lev=*/2);

    test_matmul_internal(/*m=*/128, /*n=*/128, /*k=*/256, /*bs_a=*/1,
                         /*bs_b=*/1, /*split_k=*/1,
                         /*gran_lev=*/2);

    test_matmul_internal(/*m=*/128, /*n=*/4096, /*k=*/1024, /*bs_a=*/1,
                         /*bs_b=*/1, /*split_k=*/1, /*gran_lev=*/2);
    return ark::unittest::SUCCESS;
}

ark::unittest::State test_matmul_gran3()
{
    test_matmul_internal(/*m=*/64, /*n=*/64, /*k=*/32, /*bs_a=*/1, /*bs_b=*/1,
                         /*split_k=*/1, /*gran_lev=*/3);
    test_matmul_internal(/*m=*/128, /*n=*/64, /*k=*/32, /*bs_a=*/1, /*bs_b=*/1,
                         /*split_k=*/1, /*gran_lev=*/3);
    test_matmul_internal(/*m=*/64, /*n=*/128, /*k=*/32, /*bs_a=*/1, /*bs_b=*/1,
                         /*split_k=*/1, /*gran_lev=*/3);
    test_matmul_internal(/*m=*/128, /*n=*/128, /*k=*/32, /*bs_a=*/1, /*bs_b=*/1,
                         /*split_k=*/1, /*gran_lev=*/3);

    test_matmul_internal(/*m=*/64, /*n=*/64, /*k=*/64, /*bs_a=*/1, /*bs_b=*/1,
                         /*split_k=*/1, /*gran_lev=*/3);
    test_matmul_internal(/*m=*/128, /*n=*/64, /*k=*/64, /*bs_a=*/1, /*bs_b=*/1,
                         /*split_k=*/1, /*gran_lev=*/3);
    test_matmul_internal(/*m=*/64, /*n=*/128, /*k=*/64, /*bs_a=*/1, /*bs_b=*/1,
                         /*split_k=*/1, /*gran_lev=*/3);
    test_matmul_internal(/*m=*/128, /*n=*/128, /*k=*/64, /*bs_a=*/1, /*bs_b=*/1,
                         /*split_k=*/1, /*gran_lev=*/3);
    test_matmul_internal(/*m=*/256, /*n=*/128, /*k=*/64, /*bs_a=*/1, /*bs_b=*/1,
                         /*split_k=*/1, /*gran_lev=*/3);

    test_matmul_internal(/*m=*/128, /*n=*/128, /*k=*/256, /*bs_a=*/1,
                         /*bs_b=*/1, /*split_k=*/1, /*gran_lev=*/3);

    test_matmul_internal(/*m=*/128, /*n=*/4096, /*k=*/1024, /*bs_a=*/1,
                         /*bs_b=*/1, /*split_k=*/1, /*gran_lev=*/3);
    return ark::unittest::SUCCESS;
}

ark::unittest::State test_matmul_split()
{
    test_matmul_internal(/*m=*/64, /*n=*/64, /*k=*/64, /*bs_a=*/1, /*bs_b=*/1,
                         /*split_k=*/2, /*gran_lev=*/2);
    test_matmul_internal(/*m=*/128, /*n=*/64, /*k=*/64, /*bs_a=*/1, /*bs_b=*/1,
                         /*split_k=*/2, /*gran_lev=*/2);
    test_matmul_internal(/*m=*/64, /*n=*/128, /*k=*/64, /*bs_a=*/1, /*bs_b=*/1,
                         /*split_k=*/2, /*gran_lev=*/2);
    test_matmul_internal(/*m=*/128, /*n=*/128, /*k=*/64, /*bs_a=*/1, /*bs_b=*/1,
                         /*split_k=*/2, /*gran_lev=*/2);
    test_matmul_internal(/*m=*/128, /*n=*/128, /*k=*/256, /*bs_a=*/1,
                         /*bs_b=*/1, /*split_k=*/2, /*gran_lev=*/2);
    test_matmul_internal(/*m=*/128, /*n=*/4096, /*k=*/1024, /*bs_a=*/1,
                         /*bs_b=*/1, /*split_k=*/2, /*gran_lev=*/2);

    test_matmul_internal(/*m=*/64, /*n=*/64, /*k=*/128, /*bs_a=*/1, /*bs_b=*/1,
                         /*split_k=*/4, /*gran_lev=*/2);
    test_matmul_internal(/*m=*/128, /*n=*/64, /*k=*/128, /*bs_a=*/1, /*bs_b=*/1,
                         /*split_k=*/4, /*gran_lev=*/2);
    test_matmul_internal(/*m=*/64, /*n=*/128, /*k=*/128, /*bs_a=*/1, /*bs_b=*/1,
                         /*split_k=*/4, /*gran_lev=*/2);
    test_matmul_internal(/*m=*/128, /*n=*/128, /*k=*/128, /*bs_a=*/1,
                         /*bs_b=*/1, /*split_k=*/4, /*gran_lev=*/2);
    test_matmul_internal(/*m=*/128, /*n=*/128, /*k=*/256, /*bs_a=*/1,
                         /*bs_b=*/1, /*split_k=*/4, /*gran_lev=*/2);
    test_matmul_internal(/*m=*/128, /*n=*/4096, /*k=*/1024, /*bs_a=*/1,
                         /*bs_b=*/1, /*split_k=*/4, /*gran_lev=*/2);

    test_matmul_internal(/*m=*/128, /*n=*/4096, /*k=*/1024, /*bs_a=*/1,
                         /*bs_b=*/1, /*split_k=*/4, /*gran_lev=*/0);
    test_matmul_internal(/*m=*/128, /*n=*/4096, /*k=*/1024, /*bs_a=*/1,
                         /*bs_b=*/1, /*split_k=*/4, /*gran_lev=*/1);
    test_matmul_internal(/*m=*/128, /*n=*/4096, /*k=*/1024, /*bs_a=*/1,
                         /*bs_b=*/1, /*split_k=*/4, /*gran_lev=*/2);

    test_matmul_internal(/*m=*/128, /*n=*/4096, /*k=*/1024, /*bs_a=*/1,
                         /*bs_b=*/1, /*split_k=*/3, /*gran_lev=*/0);
    test_matmul_internal(/*m=*/128, /*n=*/4096, /*k=*/1024, /*bs_a=*/1,
                         /*bs_b=*/1, /*split_k=*/3, /*gran_lev=*/1);
    test_matmul_internal(/*m=*/128, /*n=*/4096, /*k=*/1024, /*bs_a=*/1,
                         /*bs_b=*/1, /*split_k=*/3, /*gran_lev=*/2);
    test_matmul_internal(/*m=*/128, /*n=*/4096, /*k=*/1024, /*bs_a=*/1,
                         /*bs_b=*/1, /*split_k=*/3, /*gran_lev=*/2);

    test_matmul_internal(/*m=*/128, /*n=*/4096, /*k=*/1024, /*bs_a=*/1,
                         /*bs_b=*/1, /*split_k=*/5, /*gran_lev=*/0);
    test_matmul_internal(/*m=*/128, /*n=*/4096, /*k=*/1024, /*bs_a=*/1,
                         /*bs_b=*/1, /*split_k=*/5, /*gran_lev=*/1);
    test_matmul_internal(/*m=*/128, /*n=*/4096, /*k=*/1024, /*bs_a=*/1,
                         /*bs_b=*/1, /*split_k=*/5, /*gran_lev=*/2);
    test_matmul_internal(/*m=*/128, /*n=*/4096, /*k=*/1024, /*bs_a=*/1,
                         /*bs_b=*/1, /*split_k=*/5, /*gran_lev=*/2);

    test_matmul_internal(/*m=*/128, /*n=*/4096, /*k=*/1024, /*bs_a=*/1,
                         /*bs_b=*/1, /*split_k=*/6, /*gran_lev=*/0);
    test_matmul_internal(/*m=*/128, /*n=*/4096, /*k=*/1024, /*bs_a=*/1,
                         /*bs_b=*/1, /*split_k=*/6, /*gran_lev=*/1);
    test_matmul_internal(/*m=*/128, /*n=*/4096, /*k=*/1024, /*bs_a=*/1,
                         /*bs_b=*/1, /*split_k=*/6, /*gran_lev=*/2);
    test_matmul_internal(/*m=*/128, /*n=*/4096, /*k=*/1024, /*bs_a=*/1,
                         /*bs_b=*/1, /*split_k=*/6, /*gran_lev=*/2);

    test_matmul_internal(/*m=*/128, /*n=*/4096, /*k=*/1024, /*bs_a=*/1,
                         /*bs_b=*/1, /*split_k=*/7, /*gran_lev=*/0);
    test_matmul_internal(/*m=*/128, /*n=*/4096, /*k=*/1024, /*bs_a=*/1,
                         /*bs_b=*/1, /*split_k=*/7, /*gran_lev=*/1);
    test_matmul_internal(/*m=*/128, /*n=*/4096, /*k=*/1024, /*bs_a=*/1,
                         /*bs_b=*/1, /*split_k=*/7, /*gran_lev=*/2);
    test_matmul_internal(/*m=*/128, /*n=*/4096, /*k=*/1024, /*bs_a=*/1,
                         /*bs_b=*/1, /*split_k=*/7, /*gran_lev=*/2);

    test_matmul_internal(/*m=*/128, /*n=*/4096, /*k=*/1024, /*bs_a=*/1,
                         /*bs_b=*/1, /*split_k=*/8, /*gran_lev=*/0);
    test_matmul_internal(/*m=*/128, /*n=*/4096, /*k=*/1024, /*bs_a=*/1,
                         /*bs_b=*/1, /*split_k=*/8, /*gran_lev=*/1);
    test_matmul_internal(/*m=*/128, /*n=*/4096, /*k=*/1024, /*bs_a=*/1,
                         /*bs_b=*/1, /*split_k=*/8, /*gran_lev=*/2);
    test_matmul_internal(/*m=*/128, /*n=*/4096, /*k=*/1024, /*bs_a=*/1,
                         /*bs_b=*/1, /*split_k=*/8, /*gran_lev=*/2);

    return ark::unittest::SUCCESS;
}

ark::unittest::State test_matmul_perf()
{
    test_matmul_internal(/*m=*/64, /*n=*/64, /*k=*/64, /*bs_a=*/1, /*bs_b=*/1,
                         /*split_k=*/1, /*gran_lev=*/-1, /*iter=*/1000);
    test_matmul_internal(/*m=*/64, /*n=*/128, /*k=*/64, /*bs_a=*/1, /*bs_b=*/1,
                         /*split_k=*/1, /*gran_lev=*/-1, /*iter=*/1000);
    test_matmul_internal(/*m=*/128, /*n=*/64, /*k=*/64, /*bs_a=*/1, /*bs_b=*/1,
                         /*split_k=*/1, /*gran_lev=*/-1, /*iter=*/1000);
    test_matmul_internal(/*m=*/128, /*n=*/128, /*k=*/64, /*bs_a=*/1, /*bs_b=*/1,
                         /*split_k=*/1, /*gran_lev=*/-1, /*iter=*/1000);
    test_matmul_internal(/*m=*/256, /*n=*/128, /*k=*/64, /*bs_a=*/1, /*bs_b=*/1,
                         /*split_k=*/1, /*gran_lev=*/-1, /*iter=*/1000);
    return ark::unittest::SUCCESS;
}

int main()
{
    ark::init();
    UNITTEST(test_matmul_gran0);
    UNITTEST(test_matmul_gran1);
    UNITTEST(test_matmul_gran2);
    // UNITTEST(test_matmul_gran3);
    UNITTEST(test_matmul_split);
    UNITTEST(test_matmul_perf);
    return ark::unittest::SUCCESS;
}
