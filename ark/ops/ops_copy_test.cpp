// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include <cmath>

#include "ark/model.hpp"
#include "ops_test_common.hpp"
#include "unittest/unittest_utils.h"

template <typename T>
void baseline_copy(std::vector<void *> &outputs,
                   const std::vector<ark::Dims> &output_shapes,
                   const std::vector<void *> &inputs,
                   const std::vector<ark::Dims> &input_shapes, int) {
    T *out = static_cast<T *>(outputs[0]);
    T *input = static_cast<T *>(inputs[0]);
    ark::Dims osh = output_shapes[0].dims4();
    ark::Dims ish = input_shapes[0].dims4();
    for (auto i = 0; i < osh[0]; ++i) {
        for (auto j = 0; j < osh[1]; ++j) {
            for (auto k = 0; k < osh[2]; ++k) {
                for (auto l = 0; l < osh[3]; ++l) {
                    auto idx = i * osh[1] * osh[2] * osh[3] +
                               j * osh[2] * osh[3] + k * osh[3] + l;
                    auto i_i = i % ish[0];
                    auto i_j = j % ish[1];
                    auto i_k = k % ish[2];
                    auto i_l = l % ish[3];
                    auto in_idx = i_i * ish[1] * ish[2] * ish[3] +
                                  i_j * ish[2] * ish[3] + i_k * ish[3] + i_l;
                    out[idx] = input[in_idx];
                }
            }
        }
    }
};

ark::unittest::State test_copy_fp32() {
    ark::Model m;
    ark::ModelTensorRef t = m.tensor(ark::Dims(4, 2, 1024), ark::FP32);
    ark::ModelTensorRef out = m.copy(t);

    auto result =
        ark::op_test("copy_fp32", m, {t}, {out}, baseline_copy<float>);
    UNITTEST_LOG(result);
    UNITTEST_EQ(result.max_diff[0], 0.0f);
    return ark::unittest::SUCCESS;
}

ark::unittest::State test_copy_fp16() {
    ark::Model m;
    ark::ModelTensorRef t = m.tensor(ark::Dims(4, 2, 1024), ark::FP16);
    ark::ModelTensorRef out = m.copy(t);

    auto result =
        ark::op_test("copy_fp16", m, {t}, {out}, baseline_copy<ark::half_t>);
    UNITTEST_LOG(result);
    UNITTEST_EQ(result.max_diff[0], 0.0f);
    return ark::unittest::SUCCESS;
}

ark::unittest::State test_copy_bf16() {
    ark::Model m;
    ark::ModelTensorRef t = m.tensor(ark::Dims(4, 2, 1024), ark::BF16);
    ark::ModelTensorRef out = m.copy(t);

    auto result = ark::op_test("copy_bf16", m, {t}, {out},
                               baseline_copy<ark::bfloat16_t>);
    UNITTEST_LOG(result);
    UNITTEST_EQ(result.max_diff[0], 0.0f);
    return ark::unittest::SUCCESS;
}

ark::unittest::State test_copy_fp32_expand() {
    ark::Model m;
    ark::ModelTensorRef t = m.tensor(ark::Dims(4, 1, 1024), ark::FP32);
    ark::ModelTensorRef out = m.tensor(ark::Dims(4, 3, 1024), ark::FP32);
    m.copy(t, out);

    auto result =
        ark::op_test("copy_fp32_expand", m, {t}, {out}, baseline_copy<float>);
    UNITTEST_LOG(result);
    UNITTEST_EQ(result.max_diff[0], 0.0f);
    return ark::unittest::SUCCESS;
}

ark::unittest::State test_copy_invalid() {
    {
        ark::Model m;
        ark::ModelTensorRef t = m.tensor(ark::Dims(4, 1, 1024), ark::FP32);
        ark::ModelTensorRef out = m.tensor(ark::Dims(4, 3, 1024), ark::FP16);
        UNITTEST_THROW(m.copy(t, out), ark::InvalidUsageError);
    }
    {
        ark::Model m;
        ark::ModelTensorRef t = m.tensor(ark::Dims(4, 1, 1024), ark::FP32);
        ark::ModelTensorRef out = m.tensor(ark::Dims(1, 3, 1024), ark::FP16);
        UNITTEST_THROW(m.copy(t, out), ark::InvalidUsageError);
    }
    return ark::unittest::SUCCESS;
}

int main() {
    ark::init();
    UNITTEST(test_copy_fp32);
    UNITTEST(test_copy_fp16);
    UNITTEST(test_copy_bf16);
    UNITTEST(test_copy_fp32_expand);
    UNITTEST(test_copy_invalid);
    return ark::unittest::SUCCESS;
}
