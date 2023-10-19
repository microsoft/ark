// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include <cmath>

#include "include/ark.h"
#include "ops_test_common.h"
#include "unittest/unittest_utils.h"

float relu(float x) { return x > 0 ? x : 0; }

template <typename T>
void baseline_relu(std::vector<void *> &outputs,
                   const std::vector<ark::Dims> &output_shapes,
                   const std::vector<void *> &inputs,
                   const std::vector<ark::Dims> &) {
    T *out = static_cast<T *>(outputs[0]);
    T *input = static_cast<T *>(inputs[0]);
    ark::Dims osh = output_shapes[0];
    for (ark::DimType i = 0; i < osh.size(); ++i) {
        out[i] = relu(input[i]);
    }
};

ark::unittest::State test_relu_fp32() {
    ark::Model m;
    ark::Tensor *t = m.tensor(ark::Dims(4, 2, 1024), ark::FP32);
    ark::Tensor *out = m.relu(t);

    auto result =
        ark::op_test("relu_fp32", m, {t}, {out}, baseline_relu<float>);
    UNITTEST_LOG(result);
    UNITTEST_EQ(result.max_diff[0], 0.0f);
    return ark::unittest::SUCCESS;
}

ark::unittest::State test_relu_fp16() {
    ark::Model m;
    ark::Tensor *t = m.tensor(ark::Dims(4, 2, 1024), ark::FP16);
    ark::Tensor *out = m.relu(t);

    auto result =
        ark::op_test("relu_fp16", m, {t}, {out}, baseline_relu<ark::half_t>);
    UNITTEST_LOG(result);
    UNITTEST_EQ(result.max_diff[0], 0.0f);
    return ark::unittest::SUCCESS;
}

ark::unittest::State test_relu_bf16() {
    ark::Model m;
    ark::Tensor *t = m.tensor(ark::Dims(4, 2, 1024), ark::BF16);
    ark::Tensor *out = m.relu(t);

    auto result = ark::op_test("relu_bf16", m, {t}, {out},
                               baseline_relu<ark::bfloat16_t>);
    UNITTEST_LOG(result);
    UNITTEST_EQ(result.max_diff[0], 0.0f);
    return ark::unittest::SUCCESS;
}

int main() {
    ark::init();
    UNITTEST(test_relu_fp32);
    UNITTEST(test_relu_fp16);
    UNITTEST(test_relu_bf16);
    return ark::unittest::SUCCESS;
}
