// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "ark/gpu/gpu_kernel.h"
#include "ark/include/ark.h"
#include "ark/include/ark_utils.h"
#include "ark/logging.h"
#include "ark/unittest/unittest_utils.h"

using namespace std;

// m,n,k: Problem size. CAUTION: `m` and `n` are assumed to be multiple of 16.
// bs_a: Batch size of left-side matrix.
// bs_b: Batch size of right-side matrix.
// is_relu: Use ReLU activation if true.
// iter: Number of iterations.
void test_matmul_internal(unsigned int m, unsigned int n, unsigned int k,
                          unsigned int bs_a, unsigned int bs_b,
                          bool is_relu = false, unsigned int iter = 1)
{
    assert(bs_a == bs_b || bs_a == 1 || bs_b == 1);
    unsigned int bs_res = bs_a > bs_b ? bs_a : bs_b;

    ark::GpuMgr *mgr = ark::get_gpu_mgr(0);
    ark::GpuMgrCtx *ctx = mgr->create_context("test_simple_matmul_nt", 0, 1);

    size_t buf_a_sz = (size_t)bs_a * (size_t)m * (size_t)k * sizeof(half_t);
    size_t buf_b_sz = (size_t)bs_b * (size_t)k * (size_t)n * sizeof(half_t);
    size_t buf_res_sz = (size_t)bs_res * (size_t)m * (size_t)n * sizeof(half_t);

    // Reserved GPU buffers for execution of a manually written kernel,
    // `simple_matmul_nt`.
    ark::GpuBuf *buf_a = ctx->mem_alloc(buf_a_sz);
    ark::GpuBuf *buf_b = ctx->mem_alloc(buf_b_sz);
    ark::GpuBuf *buf_gt = ctx->mem_alloc(buf_res_sz);
    // ark::GpuBuf *buf_res = ctx->mem_alloc(buf_res_sz);

    ctx->freeze();

    // Define `simple_matmul_nt` kernel to generate the ground truth.
    ark::GpuKernel gk{"simple_matmul_nt",
                      {get_kernel_code("simple_matmul_nt")},
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
    auto data_a = rand_halfs(buf_a_sz / sizeof(half_t), 0.001);
    auto data_b = rand_halfs(buf_b_sz / sizeof(half_t), 0.001);
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
    ark::Tensor *tns_res = model.matmul(tns_a, tns_b, nullptr, 1, false, false,
                                        is_relu, "matmul", 3);

    mgr->destroy_context(ctx);

    //
    ark::Executor exe{0, 0, 1, model, "test_matmul_nt"};
    exe.compile();

    // Get the auto-scheduled buffers.
    ark::GpuBuf *buf_tns_a = exe.get_gpu_buf(tns_a);
    ark::GpuBuf *buf_tns_b = exe.get_gpu_buf(tns_b);
    ark::GpuBuf *buf_tns_res = exe.get_gpu_buf(tns_res);

    UNITTEST_NE(buf_tns_a, (ark::GpuBuf *)nullptr);
    UNITTEST_NE(buf_tns_b, (ark::GpuBuf *)nullptr);

    // Set data.
    ark::gpu_memcpy(buf_tns_a, data_a.get(), buf_a_sz);
    ark::gpu_memcpy(buf_tns_b, data_b.get(), buf_b_sz);

    exe.launch();
    exe.run(iter);
    float elapsed = exe.stop();

    // Copy results of the loop kernel routine into CPU memory.
    void *res = malloc(buf_res_sz);
    UNITTEST_NE(res, (void *)nullptr);
    ark::gpu_memcpy(res, buf_tns_res, buf_res_sz);

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
    //     ((half_t *)gt)[i] = half_t(temp);
    // }

    // Compare results with the ground truth.
    auto p = cmp_matrix((half_t *)gt, (half_t *)res, m, n);
    float max_err = p.second;
    LOG(ark::INFO, "matmul:", m, 'x', n, 'x', k, "(relu=", is_relu, ") ",
        setprecision(4), " mse ", p.first, " max_err ", max_err * 100, "%",
        " elapsed ", elapsed, "ms iter ", iter);

    free(res);
    free(gt);

    UNITTEST_EQ(max_err, 0.0);
}

// bs_a: Batch size of left-side matrix.
// bs_b: Batch size of right-side matrix.
void test_matmul_split_internal(unsigned int m, unsigned int n, unsigned int k,
                                unsigned int bs_a, unsigned int bs_b)
{
    assert(bs_a == bs_b || bs_a == 1 || bs_b == 1);
    // unsigned int bs_res = bs_a > bs_b ? bs_a : bs_b;

    // Context ctx;
    // Launcher *ln_1 = ctx.create_launcher(0);
    // Launcher *ln_2 = ctx.create_launcher(1);

    // size_t buf_a_sz = (size_t)bs_a * (size_t)m * (size_t)k * sizeof(half_t);
    // size_t buf_b_sz = (size_t)bs_b * (size_t)k * (size_t)n * sizeof(half_t);
    // size_t buf_res_sz = (size_t)bs_res * (size_t)m * (size_t)n *
    // sizeof(half_t);

    // // Declare an equivalent matmul using Model APIs.
    // Model model_1;
    // Tensor *tns_a_1 = model_1.tensor({1, m, 1, k}, ark::FP16);
    // Tensor *tns_b_1 = model_1.tensor({1, n, 1, k}, ark::FP16);
    // Tensor *tns_res_1 = model_1.matmul(tns_a_1, tns_b_1, nullptr, 1, false,
    // true, 3);

    // Model model_2;
    // Tensor *tns_a_2 = model_2.tensor({1, m, 1, k}, ark::FP16);
    // Tensor *tns_b_2 = model_2.tensor({1, n, 1, k}, ark::FP16);
    // Tensor *tns_res_2 = model_2.matmul(tns_a_2, tns_b_2, nullptr, 1, false,
    // true);

    // // Define an auto-scheduled loop kernel routine for the declared model.
    // Routine *ru_loop_1 = ln_1->create_routine();
    // ln_1->gpu_loop_kernel(ru_loop_1, "test_kernel_matmul_loop_1", model_1);

    // Routine *ru_loop_2 = ln_2->create_routine();
    // ln_2->gpu_loop_kernel(ru_loop_2, "test_kernel_matmul_loop_2", model_2);

    // // Get the auto-scheduled buffers.
    // GpuBuf *buf_tns_a_1 = ln_1->get_gpu_buf(tns_a_1);
    // GpuBuf *buf_tns_b_1 = ln_1->get_gpu_buf(tns_b_1);
    // GpuBuf *buf_tns_res_1 = ln_1->get_gpu_buf(tns_res_1);

    // GpuBuf *buf_tns_a_2 = ln_2->get_gpu_buf(tns_a_2);
    // GpuBuf *buf_tns_b_2 = ln_2->get_gpu_buf(tns_b_2);
    // GpuBuf *buf_tns_res_2 = ln_2->get_gpu_buf(tns_res_2);

    // UNITTEST_NE(buf_tns_a_1, (GpuBuf *)nullptr);
    // UNITTEST_NE(buf_tns_b_1, (GpuBuf *)nullptr);
    // UNITTEST_NE(buf_tns_res_1, (GpuBuf *)nullptr);

    // UNITTEST_NE(buf_tns_a_2, (GpuBuf *)nullptr);
    // UNITTEST_NE(buf_tns_b_2, (GpuBuf *)nullptr);
    // UNITTEST_NE(buf_tns_res_2, (GpuBuf *)nullptr);

    // // Freeze the context.
    // ctx.freeze();
    // {
    //     // Generate random data for tests.
    //     ark::srand();
    //     auto data_a = rand_halfs(buf_a_sz / sizeof(half_t), 0.5);
    //     auto data_b = rand_halfs(buf_b_sz / sizeof(half_t), 0.5);
    //     ark::memcpy(buf_tns_a_1, data_a.get(), buf_a_sz);
    //     ark::memcpy(buf_tns_b_1, data_b.get(), buf_b_sz);
    //     ark::memcpy(buf_tns_a_2, data_a.get(), buf_a_sz);
    //     ark::memcpy(buf_tns_b_2, data_b.get(), buf_b_sz);
    // }

    // // Launch the loop kernel routine.
    // ctx.launch({ru_loop_1});
    // ctx.run(1);
    // ctx.stop();

    // ctx.launch({ru_loop_2});
    // ctx.run(1);
    // ctx.stop();

    // // Copy results of the loop kernel routine into CPU memory.
    // void *res_1 = malloc(buf_res_sz);
    // UNITTEST_NE(res_1, (void *)nullptr);
    // ark::memcpy(res_1, buf_tns_res_1, buf_res_sz);

    // void *res_2 = malloc(buf_res_sz);
    // UNITTEST_NE(res_2, (void *)nullptr);
    // ark::memcpy(res_2, buf_tns_res_2, buf_res_sz);

    // // Compare results with the ground truth.
    // stringstream ss;
    // ss << "test_matmul_split:" << m << 'x' << n << 'x' << k;
    // test_matrix_cmp(ss.str(), res_2, res_1, m, n, k, sizeof(half_t), false);
    // free(res_1);
    // free(res_2);
}

ark::unittest::State test_matmul()
{
    test_matmul_internal(64, 64, 32, 1, 1);
    test_matmul_internal(128, 64, 32, 1, 1);
    test_matmul_internal(64, 128, 32, 1, 1);
    test_matmul_internal(128, 128, 32, 1, 1);

    test_matmul_internal(64, 64, 64, 1, 1);
    test_matmul_internal(128, 64, 64, 1, 1);
    test_matmul_internal(64, 128, 64, 1, 1);
    test_matmul_internal(128, 128, 64, 1, 1);
    test_matmul_internal(256, 128, 64, 1, 1);

    test_matmul_internal(128, 128, 256, 1, 1);

    test_matmul_internal(128, 4096, 1024, 1, 1);
    return ark::unittest::SUCCESS;
}

ark::unittest::State test_matmul_relu()
{
    test_matmul_internal(128, 4096, 1024, 1, 1, true);
    return ark::unittest::SUCCESS;
}

ark::unittest::State test_matmul_split()
{
    test_matmul_split_internal(64, 64, 1, 1, 1);
    test_matmul_split_internal(64, 64, 32, 1, 1);

    // Indivisible-sized splitting
    for (int k = 33; k < 1024; k += 33) {
        test_matmul_split_internal(64, 64, k, 1, 1);
    }
    return ark::unittest::SUCCESS;
}

ark::unittest::State test_matmul_perf()
{
    test_matmul_internal(64, 64, 64, 1, 1, false, 1000);
    test_matmul_internal(64, 128, 64, 1, 1, false, 1000);
    test_matmul_internal(128, 64, 64, 1, 1, false, 1000);
    test_matmul_internal(128, 128, 64, 1, 1, false, 1000);
    test_matmul_internal(256, 128, 64, 1, 1, false, 1000);
    return ark::unittest::SUCCESS;
}

int main()
{
    ark::init();
    UNITTEST(test_matmul);
    UNITTEST(test_matmul_relu);
    // UNITTEST(test_matmul_split);
    UNITTEST(test_matmul_perf);
    return ark::unittest::SUCCESS;
}
