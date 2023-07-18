// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "ops_test_common.h"
#include "gpu/gpu_kernel.h"
#include "include/ark_utils.h"
#include "logging.h"
#include "unittest/unittest_utils.h"

using namespace std;

// op_name: "add", "mul"
template <typename T>
void test_bcast_internal(string op_name, ark::TensorType type, ark::DimType bs,
                         ark::DimType n, ark::DimType m, bool overwrite)
{
    string type_name;
    if (type == ark::FP32) {
        type_name = "fp32";
    } else if (type == ark::FP16) {
        type_name = "fp16";
    } else {
        UNITTEST_FEXIT("Unsupported tensor type:", type);
    }
    string kernel_name = "simple_" + op_name + "_" + type_name;

    ark::GpuMgr *mgr = ark::get_gpu_mgr(0);
    ark::GpuMgrCtx *ctx = mgr->create_context("test_simple_" + op_name, 0, 1);

    ark::DimType len = m * n;
    ark::GpuBuf *buf_a = ctx->mem_alloc(bs * len * sizeof(T));
    ark::GpuBuf *buf_b = ctx->mem_alloc(len * sizeof(T));
    ark::GpuBuf *buf_c = ctx->mem_alloc(bs * len * sizeof(T));

    ctx->freeze();

    ark::GpuKernel gk{kernel_name,
                      {ark::unittest::get_kernel_code("simple_" + op_name)},
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
    auto data_a = ark::utils::rand_array<T>(bs * len, 0.01);
    auto data_b = ark::utils::rand_array<T>(len, 0.01);
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
    ark::Tensor *tns_c = nullptr;
    if (op_name == "add") {
        if (overwrite) {
            tns_c = model.add(tns_a, tns_b, tns_a);
        } else {
            tns_c = model.add(tns_a, tns_b);
        }
    } else if (op_name == "mul") {
        if (overwrite) {
            tns_c = model.mul(tns_a, tns_b, tns_a);
        } else {
            tns_c = model.mul(tns_a, tns_b);
        }
    }
    UNITTEST_NE(tns_c, (ark::Tensor *)nullptr);

    //
    ark::Executor exe{0, 0, 1, model, "test_" + op_name + "_" + type_name};
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
    std::pair<float, float> p =
        ark::utils::tensor_compare(gt, res, tns_c->shape, true);
    float max_err = p.second;

    if (overwrite) {
        exe.tensor_memcpy(res, tns_a, bs * len * sizeof(T));
        p = ark::utils::tensor_compare(gt, res, tns_a->shape, true);
        max_err = std::max(max_err, p.second);
    }

    LOG(ark::INFO, op_name, ":", n, 'x', m, ",", type_name, ",bs=", bs,
        ",overwrite=", overwrite, setprecision(4), " mse ", p.first,
        " max_err ", max_err * 100, "%");

    free(res);
    free(gt);

    UNITTEST_EQ(max_err, 0.0);
}

void test_bcast_fp32(string op_name, ark::DimType bs, ark::DimType n,
                     ark::DimType m, bool overwrite)
{
    test_bcast_internal<float>(op_name, ark::FP32, bs, n, m, overwrite);
}

void test_bcast_fp16(string op_name, ark::DimType bs, ark::DimType n,
                     ark::DimType m, bool overwrite)
{
    test_bcast_internal<ark::half_t>(op_name, ark::FP16, bs, n, m, overwrite);
}
