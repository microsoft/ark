// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include <cmath>

#include "include/ark.h"
#include "ops_test_common.h"
#include "unittest/unittest_utils.h"

using namespace std;

template <typename T>
void baseline_layernorm(std::vector<void *> &outputs,
                        const std::vector<ark::Dims> &output_shapes,
                        const std::vector<void *> &inputs,
                        const std::vector<ark::Dims> &input_shapes) {
    T *out = static_cast<T *>(outputs[0]);
    T *input = static_cast<T *>(inputs[0]);

    ark::Dims osh = output_shapes[0].dims4();
    ark::Dims ish = input_shapes[0].dims4();

    for (ark::DimType n = 0; n < ish[0]; ++n) {
        for (ark::DimType c = 0; c < ish[1]; ++c) {
            for (ark::DimType h = 0; h < ish[2]; ++h) {
                float mean = 0;
                for (ark::DimType w = 0; w < ish[3]; ++w) {
                    float inval =
                        float(input[n * ish[1] * ish[2] * ish[3] +
                                    c * ish[2] * ish[3] + h * ish[3] + w]);
                    mean += inval;
                }
                mean /= ish[3];
                float var = 0;
                for (ark::DimType w = 0; w < ish[3]; ++w) {
                    float inval =
                        float(input[n * ish[1] * ish[2] * ish[3] +
                                    c * ish[2] * ish[3] + h * ish[3] + w]);
                    var += (inval - mean) * (inval - mean);
                }
                var /= ish[3];
                var = 1.0f / std::sqrt(var + 1e-5f);
                for (ark::DimType w = 0; w < ish[3]; ++w) {
                    float inval =
                        float(input[n * ish[1] * ish[2] * ish[3] +
                                    c * ish[2] * ish[3] + h * ish[3] + w]);
                    out[n * osh[1] * osh[2] * osh[3] + c * osh[2] * osh[3] +
                        h * osh[3] + w] = T((inval - mean) * var);
                }
            }
        }
    }
}

ark::unittest::State test_layernorm_fp32() {
    ark::Model m;
    ark::Tensor *t = m.tensor(ark::Dims(1, 32, 32, 8192), ark::FP32);
    ark::Tensor *out = m.layernorm(t);
    auto result = ark::op_test("layernorm_fp32", m, {t}, {out},
                               baseline_layernorm<float>);
    UNITTEST_LOG(result);
    UNITTEST_TRUE(result.max_diff[0] < 1e-5f);
    return ark::unittest::SUCCESS;
}

ark::unittest::State test_layernorm_fp16() {
    ark::Model model;
    ark::Tensor *input = model.tensor(ark::Dims(1, 32, 32, 8192), ark::FP16);
    ark::Tensor *output = model.layernorm(input);
    auto result = ark::op_test("layernorm_fp16", model, {input}, {output},
                               baseline_layernorm<ark::half_t>);
    UNITTEST_LOG(result);
    return ark::unittest::SUCCESS;
}

ark::unittest::State test_layernorm_bf16() {
    ark::Model model;
    ark::Tensor *input = model.tensor(ark::Dims(1, 32, 32, 8192), ark::BF16);
    ark::Tensor *output = model.layernorm(input);
    auto result = ark::op_test("layernorm_bf16", model, {input}, {output},
                               baseline_layernorm<ark::bfloat16_t>);
    UNITTEST_LOG(result);
    return ark::unittest::SUCCESS;
}

int main() {
    ark::init();
    UNITTEST(test_layernorm_fp32);
    UNITTEST(test_layernorm_fp16);
    UNITTEST(test_layernorm_bf16);
    return ark::unittest::SUCCESS;
}
