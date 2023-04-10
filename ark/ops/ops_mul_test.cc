// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "ark/executor.h"
#include "ark/gpu/gpu_kernel.h"
#include "ark/init.h"
#include "ark/logging.h"
#include "ark/ops/ops_test_utils.h"
#include "ark/random.h"
#include "ark/unittest/unittest_utils.h"

using namespace std;

//
template <typename T>
void test_mul_internal(ark::TensorType type, unsigned int bs, unsigned int n,
                       unsigned int m)
{
    string kernel_name;
    if (type == ark::FP32) {
        kernel_name = "simple_mul_fp32";
    } else if (type == ark::FP16) {
        kernel_name = "simple_mul_fp16";
    } else {
        UNITTEST_FEXIT("Unsupported tensor type:", type);
    }

    ark::GpuMgr *mgr = ark::get_gpu_mgr(0);
    ark::GpuMgrCtx *ctx = mgr->create_context("test_simple_mul", 0, 1);

    unsigned int len = m * n;
    ark::GpuBuf *buf_a = ctx->mem_alloc(bs * len * sizeof(T));
    ark::GpuBuf *buf_b = ctx->mem_alloc(len * sizeof(T));
    ark::GpuBuf *buf_c = ctx->mem_alloc(bs * len * sizeof(T));

    ctx->freeze();

    ark::GpuKernel gk{kernel_name,
                      {get_kernel_code("simple_mul")},
                      {(unsigned int)mgr->get_gpu_info().num_sm, 1, 1},
                      {512, 1, 1},
                      0,
                      {buf_c, buf_a, buf_b},
                      {},
                      {
                          {&bs, sizeof(bs)},
                          {&len, sizeof(len)},
                      },
                      ""};
    gk.compile(mgr->get_gpu_info());
    gk.load();

    // Set data.
    ark::srand();
    auto data_a = rand_array<T>(bs * len, 0.01);
    auto data_b = rand_array<T>(len, 0.01);
    ark::gpu_memcpy(buf_a, data_a.get(), bs * len * sizeof(T));
    ark::gpu_memcpy(buf_b, data_b.get(), len * sizeof(T));

    // Run the GPU kernel.
    ark::GpuStream s = ctx->create_stream();
    int ret = gk.launch(s);
    UNITTEST_EQ(ret, 0);
    ret = ctx->sync_stream(s);
    UNITTEST_EQ(ret, 0);

    // Copy the ground truth results into CPU memory.
    T *gt = (T *)malloc(bs * len * sizeof(T));
    UNITTEST_NE(gt, (T *)nullptr);
    ark::gpu_memcpy(gt, buf_c, bs * len * sizeof(T));

    mgr->destroy_context(ctx);

    //
    ark::Model model;
    ark::Tensor *tns_a = model.tensor({bs, n, m}, type);
    ark::Tensor *tns_b = model.tensor({1, n, m}, type);
    ark::Tensor *tns_c = model.mul(tns_a, tns_b);

    //
    ark::Executor exe{0, 0, 1, model, "test_mul"};
    exe.compile();

    // Set data.
    exe.tensor_memcpy(tns_a, data_a.get(), bs * len * sizeof(T));
    exe.tensor_memcpy(tns_b, data_b.get(), len * sizeof(T));

    exe.launch();
    exe.run(1);
    exe.stop();

    // Copy results of the loop kernel routine into CPU memory.
    T *res = (T *)malloc(bs * len * sizeof(T));
    UNITTEST_NE(res, (T *)nullptr);
    exe.tensor_memcpy(res, tns_c, bs * len * sizeof(T));

    // Compare results with the ground truth.
    std::pair<float, float> p = tensor_compare(gt, res, tns_c->shape, true);
    float max_err = p.second;
    LOG(ark::INFO, "mul:", n, 'x', m, ",bs=", bs, setprecision(4), " mse ",
        p.first, " max_err ", max_err * 100, "%");

    free(res);
    free(gt);

    UNITTEST_EQ(max_err, 0.0);
}

ark::unittest::State test_mul_fp32()
{
    test_mul_internal<float>(ark::FP32, 1, 1, 2);
    test_mul_internal<float>(ark::FP32, 1, 1, 64);
    test_mul_internal<float>(ark::FP32, 1, 128, 128);
    test_mul_internal<float>(ark::FP32, 1, 1024, 512);
    test_mul_internal<float>(ark::FP32, 1, 512, 1024);
    test_mul_internal<float>(ark::FP32, 2, 1, 64);
    test_mul_internal<float>(ark::FP32, 2, 128, 128);
    test_mul_internal<float>(ark::FP32, 4, 1024, 512);
    test_mul_internal<float>(ark::FP32, 4, 512, 1024);
    return ark::unittest::SUCCESS;
}

ark::unittest::State test_mul_fp16()
{
    test_mul_internal<half_t>(ark::FP16, 1, 1, 2);
    test_mul_internal<half_t>(ark::FP16, 1, 1, 64);
    test_mul_internal<half_t>(ark::FP16, 1, 128, 128);
    test_mul_internal<half_t>(ark::FP16, 1, 1024, 512);
    test_mul_internal<half_t>(ark::FP16, 1, 512, 1024);
    test_mul_internal<half_t>(ark::FP16, 2, 1, 64);
    test_mul_internal<half_t>(ark::FP16, 2, 128, 128);
    test_mul_internal<half_t>(ark::FP16, 4, 1024, 512);
    test_mul_internal<half_t>(ark::FP16, 4, 512, 1024);
    return ark::unittest::SUCCESS;
}

int main()
{
    ark::init();
    UNITTEST(test_mul_fp32);
    UNITTEST(test_mul_fp16);
    return ark::unittest::SUCCESS;
}
