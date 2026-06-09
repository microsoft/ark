// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include <algorithm>
#include <cmath>

#include "ops_test_common.hpp"

// Baseline: row-wise softmax.
// For each row (last dim = W):
//   max_val = max(row)
//   exp_sum = sum(exp(x - max_val))
//   out[j]  = exp(x[j] - max_val) / exp_sum
template <typename T>
void baseline_softmax(std::vector<void *> &outputs,
                      const std::vector<ark::Dims> &output_shapes,
                      const std::vector<void *> &inputs,
                      const std::vector<ark::Dims> &, int) {
    T *out = static_cast<T *>(outputs[0]);
    T *input = static_cast<T *>(inputs[0]);

    ark::Dims osh = output_shapes[0];
    ark::DimType total = osh.nelems();
    ark::DimType W = osh[-1];
    ark::DimType num_rows = total / W;

    for (ark::DimType row = 0; row < num_rows; ++row) {
        T *row_in = input + row * W;
        T *row_out = out + row * W;

        // pass 1: max
        float max_val = static_cast<float>(row_in[0]);
        for (ark::DimType j = 1; j < W; ++j) {
            float v = static_cast<float>(row_in[j]);
            if (v > max_val) max_val = v;
        }

        // pass 2: exp and sum
        float exp_sum = 0;
        for (ark::DimType j = 0; j < W; ++j) {
            float e = std::exp(static_cast<float>(row_in[j]) - max_val);
            exp_sum += e;
        }

        // pass 3: normalize
        for (ark::DimType j = 0; j < W; ++j) {
            float e = std::exp(static_cast<float>(row_in[j]) - max_val);
            row_out[j] = T(e / exp_sum);
        }
    }
}

ark::unittest::State test_softmax_fp32() {
    ark::Model m;
    ark::Tensor input = m.tensor({4, 1024}, ark::FP32);
    ark::Tensor out = m.softmax(input);

    auto result = ark::op_test("softmax_fp32", m, {input}, {out},
                               baseline_softmax<float>);
    UNITTEST_LOG(result);
    UNITTEST_TRUE(result.max_diff[0] < 1e-5f);
    return ark::unittest::SUCCESS;
}

ark::unittest::State test_softmax_fp16() {
    ark::Model m;
    ark::Tensor input = m.tensor({2, 512}, ark::FP16);
    ark::Tensor out = m.softmax(input);

    auto result = ark::op_test("softmax_fp16", m, {input}, {out},
                               baseline_softmax<ark::half_t>);
    UNITTEST_LOG(result);
    UNITTEST_TRUE(result.max_diff[0] < 5e-3f);
    return ark::unittest::SUCCESS;
}

ark::unittest::State test_softmax_bf16() {
    ark::Model m;
    ark::Tensor input = m.tensor({2, 512}, ark::BF16);
    ark::Tensor out = m.softmax(input);

    auto result = ark::op_test("softmax_bf16", m, {input}, {out},
                               baseline_softmax<ark::bfloat16_t>);
    UNITTEST_LOG(result);
    UNITTEST_TRUE(result.max_diff[0] < 5e-2f);
    return ark::unittest::SUCCESS;
}

ark::unittest::State test_softmax_batch() {
    // Higher-dimensional: [B, H, S, S] — attention pattern
    ark::Model m;
    ark::Tensor input = m.tensor({2, 12, 64, 64}, ark::FP32);
    ark::Tensor out = m.softmax(input);

    auto result = ark::op_test("softmax_batch", m, {input}, {out},
                               baseline_softmax<float>);
    UNITTEST_LOG(result);
    UNITTEST_TRUE(result.max_diff[0] < 1e-5f);
    return ark::unittest::SUCCESS;
}

ark::unittest::State test_softmax_small_row() {
    // Small last dim — tests edge case where W < warp size
    ark::Model m;
    ark::Tensor input = m.tensor({8, 16}, ark::FP32);
    ark::Tensor out = m.softmax(input);

    auto result = ark::op_test("softmax_small_row", m, {input}, {out},
                               baseline_softmax<float>);
    UNITTEST_LOG(result);
    UNITTEST_TRUE(result.max_diff[0] < 1e-5f);
    return ark::unittest::SUCCESS;
}

ark::unittest::State test_softmax_non_pow2() {
    // Non-power-of-2 W — exercises boundary guards (idx_w + j < W)
    ark::Model m;
    ark::Tensor input = m.tensor({4, 127}, ark::FP32);
    ark::Tensor out = m.softmax(input);

    auto result = ark::op_test("softmax_non_pow2", m, {input}, {out},
                               baseline_softmax<float>);
    UNITTEST_LOG(result);
    UNITTEST_TRUE(result.max_diff[0] < 1e-5f);
    return ark::unittest::SUCCESS;
}

ark::unittest::State test_softmax_w1() {
    // W=1 boundary: softmax output must be exactly 1.0
    ark::Model m;
    ark::Tensor input = m.tensor({4, 1}, ark::FP32);
    ark::Tensor out = m.softmax(input);

    auto result =
        ark::op_test("softmax_w1", m, {input}, {out}, baseline_softmax<float>);
    UNITTEST_LOG(result);
    UNITTEST_TRUE(result.max_diff[0] < 1e-5f);
    return ark::unittest::SUCCESS;
}

ark::unittest::State test_softmax_invalid() {
    // Output shape mismatch
    {
        ark::Model m;
        ark::Tensor input = m.tensor({4, 1024}, ark::FP32);
        ark::Tensor bad_out = m.tensor({4, 512}, ark::FP32);  // wrong W
        UNITTEST_THROW(m.softmax(input, bad_out), ark::ModelError);
    }
    return ark::unittest::SUCCESS;
}

int main() {
    ark::init();
    UNITTEST(test_softmax_fp32);
    UNITTEST(test_softmax_fp16);
    UNITTEST(test_softmax_bf16);
    UNITTEST(test_softmax_batch);
    UNITTEST(test_softmax_small_row);
    UNITTEST(test_softmax_non_pow2);
    UNITTEST(test_softmax_w1);
    UNITTEST(test_softmax_invalid);
    return ark::unittest::SUCCESS;
}
