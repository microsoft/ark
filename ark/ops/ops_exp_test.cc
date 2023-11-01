// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include <cmath>

#include "include/ark.h"
#include "ops_test_common.h"
#include "unittest/unittest_utils.h"

template <typename T>
void baseline_exp(std::vector<void *> &outputs,
                  const std::vector<ark::Dims> &output_shapes,
                  const std::vector<void *> &inputs,
                  const std::vector<ark::Dims> &, int)
{
    T *out = static_cast<T *>(outputs[0]);
    T *input = static_cast<T *>(inputs[0]);
    ark::Dims osh = output_shapes[0];
    for (ark::DimType i = 0; i < osh.size(); ++i) {
        out[i] = std::exp(input[i]);
    }
};

ark::unittest::State test_exp_fp32() {
    ark::Model m;
    ark::Tensor *t = m.tensor(ark::Dims(4, 2, 1024), ark::FP32);
    ark::Tensor *out = m.exp(t);

    auto result = ark::op_test("exp_fp32", m, {t}, {out}, baseline_exp<float>);
    UNITTEST_LOG(result);
    UNITTEST_TRUE(result.max_diff[0] < 1e-5f);
    return ark::unittest::SUCCESS;
}

ark::unittest::State test_exp_fp16() {
    ark::Model m;
    ark::Tensor *t = m.tensor(ark::Dims(4, 2, 1024), ark::FP16);
    ark::Tensor *out = m.exp(t);

    auto result =
        ark::op_test("exp_fp16", m, {t}, {out}, baseline_exp<ark::half_t>);
    UNITTEST_LOG(result);
    UNITTEST_TRUE(result.max_diff[0] < 1e-2f);
    return ark::unittest::SUCCESS;
}

ark::unittest::State test_exp_bf16() {
    ark::Model m;
    ark::Tensor *t = m.tensor(ark::Dims(4, 2, 1024), ark::BF16);
    ark::Tensor *out = m.exp(t);

    auto result =
        ark::op_test("exp_bf16", m, {t}, {out}, baseline_exp<ark::bfloat16_t>);
    UNITTEST_LOG(result);
    UNITTEST_TRUE(result.max_diff[0] < 1e-2f);
    return ark::unittest::SUCCESS;
}

int main() {
    ark::init();
    UNITTEST(test_exp_fp32);
    UNITTEST(test_exp_fp16);
    UNITTEST(test_exp_bf16);
    return ark::unittest::SUCCESS;
}
