// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include <cmath>

#include "include/ark.h"
#include "include/ark_utils.h"
#include "ops_test_common.h"
#include "unittest/unittest_utils.h"

float gelu(float x) {
    return 0.5 * x * (1 + tanh(sqrt(2 / M_PI) * (x + 0.044715 * pow(x, 3))));
}

template <typename T>
void baseline_gelu(std::vector<void *> &outputs,
                   const std::vector<ark::Dims> &output_shapes,
                   const std::vector<void *> &inputs,
                   const std::vector<ark::Dims> &, int) {
    T *out = static_cast<T *>(outputs[0]);
    T *input = static_cast<T *>(inputs[0]);
    ark::Dims osh = output_shapes[0];
    for (ark::DimType i = 0; i < osh.size(); ++i) {
        out[i] = gelu(input[i]);
    }
};

ark::unittest::State test_gelu_fp32() {
    ark::Model m;
    ark::Tensor *t = m.tensor(ark::Dims(4, 2, 1024), ark::FP32);
    ark::Tensor *out = m.gelu(t);

    auto result =
        ark::op_test("gelu_fp32", m, {t}, {out}, baseline_gelu<float>);
    UNITTEST_LOG(result);
    UNITTEST_TRUE(result.max_diff[0] < 1e-6f);
    return ark::unittest::SUCCESS;
}

ark::unittest::State test_gelu_bf16() {
    ark::Model m;
    ark::Tensor *t = m.tensor(ark::Dims(4, 2, 1024), ark::BF16);
    ark::Tensor *out = m.gelu(t);

    auto result = ark::op_test("gelu_bf16", m, {t}, {out},
                               baseline_gelu<ark::bfloat16_t>);
    UNITTEST_LOG(result);
    UNITTEST_TRUE(result.max_diff[0] < 1e-6f);
    return ark::unittest::SUCCESS;
}

int main() {
    ark::init();
    UNITTEST(test_gelu_fp32);
    UNITTEST(test_gelu_bf16);
    return ark::unittest::SUCCESS;
}
