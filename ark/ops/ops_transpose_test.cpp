// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "ark/model.hpp"
#include "ark/planner.hpp"
#include "model/model_json.hpp"
#include "ops_test_common.hpp"
#include "unittest/unittest_utils.h"

#define SYNC_TEST 0

template <typename T>
void baseline_transpose_0132(std::vector<void *> &outputs,
                             const std::vector<ark::Dims> &output_shapes,
                             const std::vector<void *> &inputs,
                             const std::vector<ark::Dims> &input_shapes, int) {
    T *out = static_cast<T *>(outputs[0]);
    T *in = static_cast<T *>(inputs[0]);
    ark::Dims osh = output_shapes[0].dims4();
    ark::Dims ish = input_shapes[0].dims4();
    for (ark::DimType n = 0; n < ish[0]; ++n) {
        for (ark::DimType c = 0; c < ish[1]; ++c) {
            for (ark::DimType h = 0; h < ish[2]; ++h) {
                for (ark::DimType w = 0; w < ish[3]; ++w) {
                    // out[n][c][w][h] = in[n][c][h][w]
                    out[h + w * osh[3] + c * osh[2] * osh[3] +
                        n * osh[1] * osh[2] * osh[3]] =
                        in[w + h * ish[3] + c * ish[3] * ish[2] +
                           n * ish[3] * ish[2] * ish[1]];
                }
            }
        }
    }
};

template <typename T>
void baseline_transpose_0231(std::vector<void *> &outputs,
                             const std::vector<ark::Dims> &output_shapes,
                             const std::vector<void *> &inputs,
                             const std::vector<ark::Dims> &input_shapes, int) {
    T *out = static_cast<T *>(outputs[0]);
    T *in = static_cast<T *>(inputs[0]);
    ark::Dims osh = output_shapes[0].dims4();
    ark::Dims ish = input_shapes[0].dims4();
    for (ark::DimType n = 0; n < ish[0]; ++n) {
        for (ark::DimType c = 0; c < ish[1]; ++c) {
            for (ark::DimType h = 0; h < ish[2]; ++h) {
                for (ark::DimType w = 0; w < ish[3]; ++w) {
                    // out[n][h][w][c] = in[n][c][h][w]
                    out[c + w * osh[3] + h * osh[2] * osh[3] +
                        n * osh[1] * osh[2] * osh[3]] =
                        in[w + h * ish[3] + c * ish[3] * ish[2] +
                           n * ish[3] * ish[2] * ish[1]];
                }
            }
        }
    }
};

template <typename T>
void baseline_transpose_0213(std::vector<void *> &outputs,
                             const std::vector<ark::Dims> &output_shapes,
                             const std::vector<void *> &inputs,
                             const std::vector<ark::Dims> &input_shapes, int) {
    T *out = static_cast<T *>(outputs[0]);
    T *in = static_cast<T *>(inputs[0]);
    ark::Dims osh = output_shapes[0].dims4();
    ark::Dims ish = input_shapes[0].dims4();
    for (ark::DimType n = 0; n < ish[0]; ++n) {
        for (ark::DimType c = 0; c < ish[1]; ++c) {
            for (ark::DimType h = 0; h < ish[2]; ++h) {
                for (ark::DimType w = 0; w < ish[3]; ++w) {
                    // out[n][h][c][w] = in[n][c][h][w]
                    out[w + c * osh[3] + h * osh[2] * osh[3] +
                        n * osh[1] * osh[2] * osh[3]] =
                        in[w + h * ish[3] + c * ish[3] * ish[2] +
                           n * ish[3] * ish[2] * ish[1]];
                }
            }
        }
    }
};

template <typename T>
void baseline_transpose_sync_test(std::vector<void *> &outputs,
                                  const std::vector<ark::Dims> &,
                                  const std::vector<void *> &inputs,
                                  const std::vector<ark::Dims> &input_shapes,
                                  int) {
    T *out = static_cast<T *>(outputs[0]);
    T *in = static_cast<T *>(inputs[0]);
    ::memcpy(out, in, sizeof(T) * input_shapes[0].nelems());
};

ark::unittest::State test_transpose_0132_fp32() {
    ark::Model m;
    ark::Tensor t = m.tensor({5, 3, 32, 128}, ark::FP32);
    ark::Tensor out = m.transpose(t, {0, 1, 3, 2});

    auto result = ark::op_test("transpose_0132_fp32", m, {t}, {out},
                               baseline_transpose_0132<float>);
    UNITTEST_LOG(result);
    UNITTEST_EQ(result.max_diff[0], 0.0f);
    return ark::unittest::SUCCESS;
}

ark::unittest::State test_transpose_0132_fp16() {
    ark::Model m;
    ark::Tensor t = m.tensor({5, 3, 32, 128}, ark::FP16);
    ark::Tensor out = m.transpose(t, {0, 1, 3, 2});

    auto result = ark::op_test("transpose_0132_fp16", m, {t}, {out},
                               baseline_transpose_0132<ark::half_t>);
    UNITTEST_LOG(result);
    UNITTEST_EQ(result.max_diff[0], 0.0f);
    return ark::unittest::SUCCESS;
}

ark::unittest::State test_transpose_0132_bf16() {
    ark::Model m;
    ark::Tensor t = m.tensor({5, 3, 32, 128}, ark::BF16);
    ark::Tensor out = m.transpose(t, {0, 1, 3, 2});

    auto result = ark::op_test("transpose_0132_bf16", m, {t}, {out},
                               baseline_transpose_0132<ark::bfloat16_t>);
    UNITTEST_LOG(result);
    UNITTEST_EQ(result.max_diff[0], 0.0f);
    return ark::unittest::SUCCESS;
}

ark::unittest::State test_transpose_0231_fp32() {
    ark::Model m;
    ark::Tensor t = m.tensor({5, 3, 32, 128}, ark::FP32);
    ark::Tensor out = m.transpose(t, {0, 2, 3, 1});

    auto result = ark::op_test("transpose_0231_fp32", m, {t}, {out},
                               baseline_transpose_0231<float>);
    UNITTEST_LOG(result);
    UNITTEST_EQ(result.max_diff[0], 0.0f);
    return ark::unittest::SUCCESS;
}

ark::unittest::State test_transpose_0231_fp16() {
    ark::Model m;
    ark::Tensor t = m.tensor({5, 3, 32, 128}, ark::FP16);
    ark::Tensor out = m.transpose(t, {0, 2, 3, 1});

    auto result = ark::op_test("transpose_0231_fp16", m, {t}, {out},
                               baseline_transpose_0231<ark::half_t>);
    UNITTEST_LOG(result);
    UNITTEST_EQ(result.max_diff[0], 0.0f);
    return ark::unittest::SUCCESS;
}

ark::unittest::State test_transpose_0231_bf16() {
    ark::Model m;
    ark::Tensor t = m.tensor({5, 3, 32, 128}, ark::BF16);
    ark::Tensor out = m.transpose(t, {0, 2, 3, 1});

    auto result = ark::op_test("transpose_0231_bf16", m, {t}, {out},
                               baseline_transpose_0231<ark::bfloat16_t>);
    UNITTEST_LOG(result);
    UNITTEST_EQ(result.max_diff[0], 0.0f);
    return ark::unittest::SUCCESS;
}

ark::unittest::State test_transpose_0213_fp32() {
    ark::Model m;
    ark::Tensor t = m.tensor({5, 3, 32, 128}, ark::FP32);
    ark::Tensor out = m.transpose(t, {0, 2, 1, 3});

    auto result = ark::op_test("transpose_0213_fp32", m, {t}, {out},
                               baseline_transpose_0213<float>);
    UNITTEST_LOG(result);
    UNITTEST_EQ(result.max_diff[0], 0.0f);
    return ark::unittest::SUCCESS;
}

ark::unittest::State test_transpose_0213_fp16() {
    ark::Model m;
    ark::PlannerContext ctx(m);
    ctx.warp_range(0, 4);
    ctx.sram_range(0, 0);
    ctx.sync(false);
    ctx.config(ark::Json({{"NumWarps", 4}, {"SramBytes", 0}, {"Tile", {8, 64}}})
                   .dump());

    ark::Tensor t = m.tensor({5, 256, 32, 128}, ark::FP16);
    ark::Tensor out = m.transpose(t, {0, 2, 1, 3});

    auto result = ark::op_test("transpose_0213_fp16", m, {t}, {out},
                               baseline_transpose_0213<ark::half_t>);
    UNITTEST_LOG(result);
    UNITTEST_EQ(result.max_diff[0], 0.0f);
    return ark::unittest::SUCCESS;
}

ark::unittest::State test_transpose_0213_bf16() {
    ark::Model m;
    ark::Tensor t = m.tensor({5, 3, 32, 128}, ark::BF16);
    ark::Tensor out = m.transpose(t, {0, 2, 1, 3});

    auto result = ark::op_test("transpose_0213_bf16", m, {t}, {out},
                               baseline_transpose_0213<ark::bfloat16_t>);
    UNITTEST_LOG(result);
    UNITTEST_EQ(result.max_diff[0], 0.0f);
    return ark::unittest::SUCCESS;
}

ark::unittest::State test_transpose_sync_test() {
    ark::Model m;
    ark::PlannerContext shared_ctx(m);
    shared_ctx.warp_range(0, 4);
    shared_ctx.sram_range(0, 0);
    shared_ctx.sync(false);

    ark::Tensor in, t, out;
    in = m.tensor({1, 16, 2, 64}, ark::FP16);
    {
        ark::PlannerContext ctx(m);
        ctx.config(
            ark::Json({{"NumWarps", 4}, {"SramBytes", 0}, {"Tile", {8, 64}}})
                .dump());
        t = m.transpose(in, {0, 2, 1, 3});
    }
    {
        ark::PlannerContext ctx(m);
        ctx.config(
            ark::Json({{"NumWarps", 4}, {"SramBytes", 0}, {"Tile", {8, 1, 64}}})
                .dump());
        out = m.transpose(t, {0, 2, 1, 3});
    }

    auto result = ark::op_test("transpose_sync_test", m, {in}, {out},
                               baseline_transpose_sync_test<ark::half_t>);
    UNITTEST_LOG(result);
    UNITTEST_EQ(result.max_diff[0], 0.0f);
    return ark::unittest::SUCCESS;
}

ark::unittest::State test_transpose_invalid() {
    {
        ark::Model m;
        ark::Tensor t = m.tensor({5}, ark::FP32);
        UNITTEST_THROW(m.transpose(t, {0, 2, 3, 1}), ark::ModelError);
    }
    {
        ark::Model m;
        ark::Tensor t = m.tensor({5, 128}, ark::FP32);
        UNITTEST_THROW(m.transpose(t, {0, 2, 3, 1}), ark::ModelError);
    }
    {
        ark::Model m;
        ark::Tensor t = m.tensor({5, 128}, ark::FP32);
        UNITTEST_THROW(m.transpose(t, {0, 2}), ark::ModelError);
    }
    {
        ark::Model m;
        ark::Tensor t = m.tensor({5, 128}, ark::FP32);
        UNITTEST_THROW(m.transpose(t, {1, 1}), ark::ModelError);
    }
    return ark::unittest::SUCCESS;
}

int main() {
    ark::init();
    UNITTEST(test_transpose_0132_fp32);
    UNITTEST(test_transpose_0132_fp16);
    UNITTEST(test_transpose_0132_bf16);
    UNITTEST(test_transpose_0231_fp32);
    UNITTEST(test_transpose_0231_fp16);
    UNITTEST(test_transpose_0231_bf16);
    UNITTEST(test_transpose_0213_fp32);
    UNITTEST(test_transpose_0213_fp16);
    UNITTEST(test_transpose_0213_bf16);
#if (SYNC_TEST)
    UNITTEST(test_transpose_sync_test);
#endif
    UNITTEST(test_transpose_invalid);
    return ark::unittest::SUCCESS;
}
