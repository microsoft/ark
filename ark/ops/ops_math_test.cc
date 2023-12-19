// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include <cmath>

#include "include/ark.h"
#include "ops_test_common.h"
#include "unittest/unittest_utils.h"

template <typename T>
void baseline_sqrt(std::vector<void *> &outputs,
                   const std::vector<ark::Dims> &output_shapes,
                   const std::vector<void *> &inputs,
                   const std::vector<ark::Dims> &, int) {
    T *out = static_cast<T *>(outputs[0]);
    T *input = static_cast<T *>(inputs[0]);
    ark::Dims osh = output_shapes[0];
    for (ark::DimType i = 0; i < osh.size(); ++i) {
        out[i] = std::sqrt(input[i]);
    }
};

template <typename T>
void baseline_rsqrt(std::vector<void *> &outputs,
                    const std::vector<ark::Dims> &output_shapes,
                    const std::vector<void *> &inputs,
                    const std::vector<ark::Dims> &, int) {
    T *out = static_cast<T *>(outputs[0]);
    T *input = static_cast<T *>(inputs[0]);
    ark::Dims osh = output_shapes[0];
    for (ark::DimType i = 0; i < osh.size(); ++i) {
        out[i] = 1.0f / std::sqrt(input[i]);
    }
};

ark::unittest::State test_math_sqrt_fp32() {
    ark::Model m;
    ark::Tensor *t = m.tensor(ark::Dims(4, 2, 1024), ark::FP32);
    ark::Tensor *out = m.sqrt(t);

    auto result =
        ark::op_test("math_sqrt_fp32", m, {t}, {out}, baseline_sqrt<float>);
    UNITTEST_LOG(result);
    UNITTEST_TRUE(result.max_diff[0] < 1e-6f);
    return ark::unittest::SUCCESS;
}

ark::unittest::State test_math_sqrt_fp16_small_last_dim() {
    ark::Model m;
    ark::Tensor *t = m.tensor(ark::Dims(4, 1024, 1), ark::FP16);
    ark::Tensor *out = m.sqrt(t);

    auto result = ark::op_test("math_sqrt_fp16_small_last_dim", m, {t}, {out},
                               baseline_sqrt<ark::half_t>);
    UNITTEST_LOG(result);
    UNITTEST_TRUE(result.max_diff[0] < 1e-4f);
    return ark::unittest::SUCCESS;
}

ark::unittest::State test_math_sqrt_invalid() {
    {
        ark::Model model;
        ark::Tensor *input = model.tensor(ark::Dims(1, 3, 16, 8192), ark::FP32);
        ark::Tensor *output =
            model.tensor(ark::Dims(1, 3, 16, 8192), ark::FP16);
        UNITTEST_THROW(model.sqrt(input, output), ark::InvalidUsageError);
    }
    {
        ark::Model model;
        ark::Tensor *input = model.tensor(ark::Dims(1, 3, 16, 8192), ark::FP32);
        ark::Tensor *output =
            model.tensor(ark::Dims(1, 3, 16, 1024), ark::FP32);
        UNITTEST_THROW(model.sqrt(input, output), ark::InvalidUsageError);
    }
    return ark::unittest::SUCCESS;
}

ark::unittest::State test_math_rsqrt_fp32() {
    ark::Model m;
    ark::Tensor *t = m.tensor(ark::Dims(4, 2, 1024), ark::FP32);
    ark::Tensor *out = m.rsqrt(t);

    auto result =
        ark::op_test("math_rsqrt_fp32", m, {t}, {out}, baseline_rsqrt<float>);
    UNITTEST_LOG(result);
    UNITTEST_TRUE(result.max_diff[0] < 1e-4f);
    return ark::unittest::SUCCESS;
}

int main() {
    ark::init();
    UNITTEST(test_math_sqrt_fp32);
    UNITTEST(test_math_sqrt_fp16_small_last_dim);
    UNITTEST(test_math_sqrt_invalid);
    UNITTEST(test_math_rsqrt_fp32);
    return ark::unittest::SUCCESS;
}
