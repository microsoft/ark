// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "include/ark.h"
#include "include/ark_utils.h"
#include "ops_test_common.h"
#include "unittest/unittest_utils.h"
#include <cmath>

template <typename T>
void baseline_softmax(std::vector<void *> &outputs,
                      const std::vector<ark::Dims> &output_shapes,
                      const std::vector<void *> &inputs,
                      const std::vector<ark::Dims> &input_shapes)
{
    T *out = static_cast<T *>(outputs[0]);
    T *input = static_cast<T *>(inputs[0]);

    ark::Dims osh = output_shapes[0].dims4();
    ark::Dims ish = input_shapes[0].dims4();

    for (ark::DimType n = 0; n < ish[0]; ++n) {
        for (ark::DimType c = 0; c < ish[1]; ++c) {
            for (ark::DimType h = 0; h < ish[2]; ++h) {
                float maxval = std::numeric_limits<float>::lowest();
                for (ark::DimType w = 0; w < ish[3]; ++w) {
                    float val =
                        float(input[w + h * ish[3] + c * ish[2] * ish[3] +
                                    n * ish[1] * ish[2] * ish[3]]);
                    if (val > maxval) {
                        maxval = val;
                    }
                }
                float sum = 0;
                std::vector<float> exps(ish[3]);
                for (ark::DimType w = 0; w < ish[3]; ++w) {
                    exps[w] = std::exp(
                        float(input[w + h * ish[3] + c * ish[2] * ish[3] +
                                    n * ish[1] * ish[2] * ish[3]]) -
                        maxval);
                    sum += exps[w];
                }
                for (ark::DimType w = 0; w < ish[3]; ++w) {
                    out[w + h * osh[3] + c * osh[2] * osh[3] +
                        n * osh[1] * osh[2] * osh[3]] = T(exps[w] / sum);
                }
            }
        }
    }
};

ark::unittest::State test_softmax_fp32()
{
    ark::Model m;
    ark::Tensor *t = m.tensor(ark::Dims(64, 8192), ark::FP32);
    ark::Tensor *out = m.softmax(t);

    auto result =
        ark::op_test("softmax_fp32", m, {t}, {out}, baseline_softmax<float>);
    UNITTEST_LOG(result);
    UNITTEST_TRUE(result.max_diff[0] < 1e-7f);
    return ark::unittest::SUCCESS;
}

ark::unittest::State test_softmax_fp16()
{
    {
        ark::Model m;
        ark::Tensor *t = m.tensor(ark::Dims(64, 8192), ark::FP16);
        ark::Tensor *out = m.softmax(t);

        auto result = ark::op_test("softmax_fp16", m, {t}, {out},
                                   baseline_softmax<ark::half_t>);
        UNITTEST_LOG(result);
        UNITTEST_TRUE(result.max_diff[0] < 1e-3f);
    }
    {
        ark::Model m;
        ark::Tensor *t = m.tensor(ark::Dims(1, 32, 64, 64), ark::FP16);
        ark::Tensor *out = m.softmax(t);

        auto result = ark::op_test("softmax_fp16", m, {t}, {out},
                                   baseline_softmax<ark::half_t>);
        UNITTEST_LOG(result);
        UNITTEST_TRUE(result.max_diff[0] < 1e-3f);
    }
    return ark::unittest::SUCCESS;
}

ark::unittest::State test_softmax_fp16_padded()
{
    ark::Model m;
    ark::Tensor *t =
        m.tensor(ark::Dims(2, 4), ark::FP16, nullptr, ark::Dims(2, 8));
    ark::Tensor *out = m.softmax(t);

    auto result = ark::op_test("softmax_fp16_padded", m, {t}, {out},
                               baseline_softmax<ark::half_t>);
    UNITTEST_LOG(result);
    UNITTEST_TRUE(result.max_diff[0] < 1e-3f);
    return ark::unittest::SUCCESS;
}

ark::unittest::State test_softmax_fp16_big_magnitude()
{
    ark::Model model;
    ark::Tensor *input = model.tensor(ark::Dims(1, 32, 32, 8192), ark::FP16);
    ark::Tensor *output = model.softmax(input);

    // set input to a big magnitude
    std::vector<ark::half_t> input_data;
    for (ark::DimType i = 0; i < input->shape.size(); ++i) {
        // larger to smaller
        input_data.push_back(ark::half_t((2047 - i) % 2048 - 1024));
    }

    auto result = ark::op_test("softmax_fp16_big_magnitude", model, {input},
                               {output}, baseline_softmax<ark::half_t>);
    UNITTEST_LOG(result);
    UNITTEST_TRUE(result.max_diff[0] < 1e-5f);
    return ark::unittest::SUCCESS;
}

ark::unittest::State test_softmax_fp16_small_magnitude()
{
    ark::Model model;
    ark::Tensor *input = model.tensor(ark::Dims(1, 32, 32, 8192), ark::FP16);
    ark::Tensor *output = model.softmax(input);

    // set input to a small magnitude
    std::vector<ark::half_t> input_data;
    for (ark::DimType i = 0; i < input->shape.size(); ++i) {
        // larger to smaller
        input_data.push_back(ark::half_t(((2047 - i) % 2048 - 1024) * 1e-7f));
    }

    auto result = ark::op_test("softmax_fp16_small_magnitude", model, {input},
                               {output}, baseline_softmax<ark::half_t>);
    UNITTEST_LOG(result);
    UNITTEST_TRUE(result.max_diff[0] < 1e-5f);
    return ark::unittest::SUCCESS;
}

int main()
{
    ark::init();
    UNITTEST(test_softmax_fp32);
    UNITTEST(test_softmax_fp16);
    UNITTEST(test_softmax_fp16_padded);
    UNITTEST(test_softmax_fp16_big_magnitude);
    UNITTEST(test_softmax_fp16_small_magnitude);
    return ark::unittest::SUCCESS;
}
