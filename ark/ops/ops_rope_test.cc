// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include <assert.h>

#include "include/ark.h"
#include "include/ark_utils.h"
#include "ops_test_common.h"
#include "unittest/unittest_utils.h"

using namespace std;

template <typename T>
void baseline_rope(std::vector<void *> &outputs, const std::vector<ark::Dims> &,
                   const std::vector<void *> &inputs,
                   const std::vector<ark::Dims> &input_shapes, int) {
    T *out = static_cast<T *>(outputs[0]);
    T *input = static_cast<T *>(inputs[0]);
    T *other = static_cast<T *>(inputs[1]);

    ark::Dims ish = input_shapes[0].dims4();

    for (ark::DimType n = 0; n < ish[0]; ++n) {
        for (ark::DimType c = 0; c < ish[1]; ++c) {
            for (ark::DimType h = 0; h < ish[2]; ++h) {
                for (ark::DimType w = 0; w < ish[3]; w += 2) {
                    int idx = n * ish[1] * ish[2] * ish[3] +
                              c * ish[2] * ish[3] + h * ish[3] + w;
                    T input0 = input[idx];
                    T input1 = input[idx + 1];
                    T other0 = other[idx];
                    T other1 = other[idx + 1];
                    out[idx] = input0 * other0 - input1 * other1;
                    out[idx + 1] = input0 * other1 + input1 * other0;
                }
            }
        }
    }
}

ark::unittest::State test_rope_fp32() {
    ark::Model model;
    ark::Tensor *input = model.tensor(ark::Dims(1, 32, 32, 256), ark::FP32);
    ark::Tensor *other = model.tensor(ark::Dims(1, 32, 32, 256), ark::FP32);
    ark::Tensor *out = model.rope(input, other);
    auto result = ark::op_test("rope", model, {input, other}, {out},
                               baseline_rope<float>);
    UNITTEST_LOG(result);
    UNITTEST_TRUE(result.max_diff[0] < 1e-6f);
    return ark::unittest::SUCCESS;
}

ark::unittest::State test_rope_fp16() {
    ark::Model model;
    ark::Tensor *input = model.tensor(ark::Dims(1, 32, 32, 256), ark::FP16);
    ark::Tensor *other = model.tensor(ark::Dims(1, 32, 32, 256), ark::FP16);
    ark::Tensor *out = model.rope(input, other);
    auto result = ark::op_test("rope", model, {input, other}, {out},
                               baseline_rope<ark::half_t>);
    UNITTEST_LOG(result);
    UNITTEST_TRUE(result.max_diff[0] < 1e-3f);
    return ark::unittest::SUCCESS;
}

ark::unittest::State test_rope_bf16() {
    ark::Model model;
    ark::Tensor *input = model.tensor(ark::Dims(1, 32, 32, 256), ark::BF16);
    ark::Tensor *other = model.tensor(ark::Dims(1, 32, 32, 256), ark::BF16);
    ark::Tensor *out = model.rope(input, other);
    auto result = ark::op_test("rope", model, {input, other}, {out},
                               baseline_rope<ark::bfloat16_t>);
    UNITTEST_LOG(result);
    UNITTEST_TRUE(result.max_diff[0] < 1e-3f);
    return ark::unittest::SUCCESS;
}

int main() {
    ark::init();
    UNITTEST(test_rope_fp32);
    UNITTEST(test_rope_fp16);
    UNITTEST(test_rope_bf16);
    return ark::unittest::SUCCESS;
}
