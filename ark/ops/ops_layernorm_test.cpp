// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include <cmath>

#include "ark/model.hpp"
#include "ops_test_common.hpp"
#include "unittest/unittest_utils.h"

// Baseline: LayerNorm with affine (gamma, beta).
// For each row (last dim = W), compute:
//   mean = sum(x) / W
//   var  = sum((x - mean)^2) / W
//   out  = gamma * (x - mean) / sqrt(var + eps) + beta
// eps = 1e-5 (standard).
template <typename T>
void baseline_layernorm(std::vector<void *> &outputs,
                        const std::vector<ark::Dims> &output_shapes,
                        const std::vector<void *> &inputs,
                        const std::vector<ark::Dims> &input_shapes, int) {
    T *out = static_cast<T *>(outputs[0]);
    T *input = static_cast<T *>(inputs[0]);
    T *gamma = static_cast<T *>(inputs[1]);
    T *beta = static_cast<T *>(inputs[2]);

    ark::Dims osh = output_shapes[0];
    ark::DimType total = osh.nelems();
    ark::DimType W = osh[-1];
    ark::DimType num_rows = total / W;
    constexpr float eps = 1e-5f;

    for (ark::DimType row = 0; row < num_rows; ++row) {
        T *row_in = input + row * W;
        T *row_out = out + row * W;

        // mean
        float mean = 0;
        for (ark::DimType j = 0; j < W; ++j) {
            mean += static_cast<float>(row_in[j]);
        }
        mean /= static_cast<float>(W);

        // variance
        float var = 0;
        for (ark::DimType j = 0; j < W; ++j) {
            float diff = static_cast<float>(row_in[j]) - mean;
            var += diff * diff;
        }
        var /= static_cast<float>(W);

        float inv_std = 1.0f / std::sqrt(var + eps);

        // normalize + affine
        for (ark::DimType j = 0; j < W; ++j) {
            float normed =
                (static_cast<float>(row_in[j]) - mean) * inv_std;
            row_out[j] =
                T(static_cast<float>(gamma[j]) * normed +
                  static_cast<float>(beta[j]));
        }
    }
}

ark::unittest::State test_layernorm_fp32() {
    ark::Model m;
    ark::Tensor input = m.tensor({4, 1024}, ark::FP32);
    ark::Tensor gamma = m.tensor({1024}, ark::FP32);
    ark::Tensor beta = m.tensor({1024}, ark::FP32);
    ark::Tensor out = m.layernorm(input, gamma, beta);

    auto result = ark::op_test("layernorm_fp32", m, {input, gamma, beta},
                               {out}, baseline_layernorm<float>);
    UNITTEST_LOG(result);
    UNITTEST_TRUE(result.max_diff[0] < 1e-4f);
    return ark::unittest::SUCCESS;
}

ark::unittest::State test_layernorm_fp16() {
    ark::Model m;
    ark::Tensor input = m.tensor({2, 768}, ark::FP16);
    ark::Tensor gamma = m.tensor({768}, ark::FP16);
    ark::Tensor beta = m.tensor({768}, ark::FP16);
    ark::Tensor out = m.layernorm(input, gamma, beta);

    auto result = ark::op_test("layernorm_fp16", m, {input, gamma, beta},
                               {out}, baseline_layernorm<ark::half_t>);
    UNITTEST_LOG(result);
    UNITTEST_TRUE(result.max_diff[0] < 5e-2f);
    return ark::unittest::SUCCESS;
}

ark::unittest::State test_layernorm_bf16() {
    ark::Model m;
    ark::Tensor input = m.tensor({2, 768}, ark::BF16);
    ark::Tensor gamma = m.tensor({768}, ark::BF16);
    ark::Tensor beta = m.tensor({768}, ark::BF16);
    ark::Tensor out = m.layernorm(input, gamma, beta);

    auto result = ark::op_test("layernorm_bf16", m, {input, gamma, beta},
                               {out}, baseline_layernorm<ark::bfloat16_t>);
    UNITTEST_LOG(result);
    UNITTEST_TRUE(result.max_diff[0] < 5e-2f);
    return ark::unittest::SUCCESS;
}

ark::unittest::State test_layernorm_batch() {
    // Higher-dimensional input: [B, S, D]
    ark::Model m;
    ark::Tensor input = m.tensor({2, 8, 512}, ark::FP32);
    ark::Tensor gamma = m.tensor({512}, ark::FP32);
    ark::Tensor beta = m.tensor({512}, ark::FP32);
    ark::Tensor out = m.layernorm(input, gamma, beta);

    auto result = ark::op_test("layernorm_batch", m, {input, gamma, beta},
                               {out}, baseline_layernorm<float>);
    UNITTEST_LOG(result);
    UNITTEST_TRUE(result.max_diff[0] < 1e-4f);
    return ark::unittest::SUCCESS;
}

ark::unittest::State test_layernorm_invalid() {
    // gamma/beta shape mismatch
    {
        ark::Model m;
        ark::Tensor input = m.tensor({4, 1024}, ark::FP32);
        ark::Tensor gamma = m.tensor({512}, ark::FP32);  // wrong size
        ark::Tensor beta = m.tensor({1024}, ark::FP32);
        UNITTEST_THROW(m.layernorm(input, gamma, beta), ark::ModelError);
    }
    return ark::unittest::SUCCESS;
}

int main() {
    ark::init();
    UNITTEST(test_layernorm_fp32);
    UNITTEST(test_layernorm_fp16);
    UNITTEST(test_layernorm_bf16);
    UNITTEST(test_layernorm_batch);
    UNITTEST(test_layernorm_invalid);
    return ark::unittest::SUCCESS;
}
