// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include <cmath>

#include "ark/model.hpp"
#include "ops_test_common.hpp"
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
    for (ark::DimType i = 0; i < osh.nelems(); ++i) {
        out[i] = gelu(input[i]);
    }
};

template <typename T>
void baseline_exp(std::vector<void *> &outputs,
                  const std::vector<ark::Dims> &output_shapes,
                  const std::vector<void *> &inputs,
                  const std::vector<ark::Dims> &, int) {
    T *out = static_cast<T *>(outputs[0]);
    T *input = static_cast<T *>(inputs[0]);
    ark::Dims osh = output_shapes[0];
    for (ark::DimType i = 0; i < osh.nelems(); ++i) {
        out[i] = std::exp(input[i]);
    }
};

float relu(float x) { return x > 0 ? x : 0; }

template <typename T>
void baseline_relu(std::vector<void *> &outputs,
                   const std::vector<ark::Dims> &output_shapes,
                   const std::vector<void *> &inputs,
                   const std::vector<ark::Dims> &, int) {
    T *out = static_cast<T *>(outputs[0]);
    T *input = static_cast<T *>(inputs[0]);
    ark::Dims osh = output_shapes[0];
    for (ark::DimType i = 0; i < osh.nelems(); ++i) {
        out[i] = relu(input[i]);
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
    for (ark::DimType i = 0; i < osh.nelems(); ++i) {
        out[i] = 1.0f / std::sqrt(input[i]);
    }
};

float sigmoid(float x) { return 1 / (1 + std::exp(-x)); }

template <typename T>
void baseline_sigmoid(std::vector<void *> &outputs,
                      const std::vector<ark::Dims> &output_shapes,
                      const std::vector<void *> &inputs,
                      const std::vector<ark::Dims> &, int) {
    T *out = static_cast<T *>(outputs[0]);
    T *input = static_cast<T *>(inputs[0]);
    ark::Dims osh = output_shapes[0];
    for (ark::DimType i = 0; i < osh.nelems(); ++i) {
        out[i] = sigmoid(input[i]);
    }
};

template <typename T>
void baseline_sqrt(std::vector<void *> &outputs,
                   const std::vector<ark::Dims> &output_shapes,
                   const std::vector<void *> &inputs,
                   const std::vector<ark::Dims> &, int) {
    T *out = static_cast<T *>(outputs[0]);
    T *input = static_cast<T *>(inputs[0]);
    ark::Dims osh = output_shapes[0];
    for (ark::DimType i = 0; i < osh.nelems(); ++i) {
        out[i] = std::sqrt(input[i]);
    }
};

ark::unittest::State test_gelu_fp32() {
    ark::Model m;
    ark::Tensor t = m.tensor({4, 2, 1024}, ark::FP32);
    ark::Tensor out = m.gelu(t);

    auto result =
        ark::op_test("gelu_fp32", m, {t}, {out}, baseline_gelu<float>);
    UNITTEST_LOG(result);
    UNITTEST_TRUE(result.max_diff[0] < 1e-6f);
    return ark::unittest::SUCCESS;
}

ark::unittest::State test_gelu_bf16() {
    ark::Model m;
    ark::Tensor t = m.tensor({4, 2, 1024}, ark::BF16);
    ark::Tensor out = m.gelu(t);

    auto result = ark::op_test("gelu_bf16", m, {t}, {out},
                               baseline_gelu<ark::bfloat16_t>);
    UNITTEST_LOG(result);
    UNITTEST_TRUE(result.max_diff[0] < 1e-6f);
    return ark::unittest::SUCCESS;
}

ark::unittest::State test_gelu_invalid() {
    {
        ark::Model m;
        ark::Tensor t = m.tensor({4, 2, 1024}, ark::BF16);
        ark::Tensor out = m.tensor({4, 2, 1024}, ark::FP32);
        UNITTEST_THROW(m.gelu(t, out), ark::InvalidUsageError);
    }
    {
        ark::Model m;
        ark::Tensor t = m.tensor({4, 2, 1024}, ark::BF16);
        ark::Tensor out = m.tensor({4, 4, 1024}, ark::BF16);
        UNITTEST_THROW(m.gelu(t, out), ark::InvalidUsageError);
    }
    return ark::unittest::SUCCESS;
}

ark::unittest::State test_exp_fp32() {
    ark::Model m;
    ark::Tensor t = m.tensor({4, 2, 1024}, ark::FP32);
    ark::Tensor out = m.exp(t);

    auto result = ark::op_test("exp_fp32", m, {t}, {out}, baseline_exp<float>);
    UNITTEST_LOG(result);
    UNITTEST_TRUE(result.max_diff[0] < 1e-5f);
    return ark::unittest::SUCCESS;
}

ark::unittest::State test_exp_fp16() {
    ark::Model m;
    ark::Tensor t = m.tensor({4, 2, 1024}, ark::FP16);
    ark::Tensor out = m.exp(t);

    auto result =
        ark::op_test("exp_fp16", m, {t}, {out}, baseline_exp<ark::half_t>);
    UNITTEST_LOG(result);
    UNITTEST_TRUE(result.max_diff[0] < 1e-2f);
    return ark::unittest::SUCCESS;
}

ark::unittest::State test_exp_bf16() {
    ark::Model m;
    ark::Tensor t = m.tensor({4, 2, 1024}, ark::BF16);
    ark::Tensor out = m.exp(t);

    auto result =
        ark::op_test("exp_bf16", m, {t}, {out}, baseline_exp<ark::bfloat16_t>);
    UNITTEST_LOG(result);
    UNITTEST_TRUE(result.max_diff[0] < 1e-2f);
    return ark::unittest::SUCCESS;
}

ark::unittest::State test_exp_invalid() {
    {
        ark::Model m;
        ark::Tensor t = m.tensor({4, 2, 1024}, ark::BF16);
        ark::Tensor out = m.tensor({4, 2, 1024}, ark::FP32);
        UNITTEST_THROW(m.exp(t, out), ark::InvalidUsageError);
    }
    {
        ark::Model m;
        ark::Tensor t = m.tensor({4, 2, 1024}, ark::BF16);
        ark::Tensor out = m.tensor({4, 4, 1024}, ark::BF16);
        UNITTEST_THROW(m.exp(t, out), ark::InvalidUsageError);
    }
    return ark::unittest::SUCCESS;
}

ark::unittest::State test_relu_fp32() {
    ark::Model m;
    ark::Tensor t = m.tensor({4, 2, 1024}, ark::FP32);
    ark::Tensor out = m.relu(t);

    auto result =
        ark::op_test("relu_fp32", m, {t}, {out}, baseline_relu<float>);
    UNITTEST_LOG(result);
    UNITTEST_EQ(result.max_diff[0], 0.0f);
    return ark::unittest::SUCCESS;
}

ark::unittest::State test_relu_fp16() {
    ark::Model m;
    ark::Tensor t = m.tensor({4, 2, 1024}, ark::FP16);
    ark::Tensor out = m.relu(t);

    auto result =
        ark::op_test("relu_fp16", m, {t}, {out}, baseline_relu<ark::half_t>);
    UNITTEST_LOG(result);
    UNITTEST_EQ(result.max_diff[0], 0.0f);
    return ark::unittest::SUCCESS;
}

ark::unittest::State test_relu_bf16() {
    ark::Model m;
    ark::Tensor t = m.tensor({4, 2, 1024}, ark::BF16);
    ark::Tensor out = m.relu(t);

    auto result = ark::op_test("relu_bf16", m, {t}, {out},
                               baseline_relu<ark::bfloat16_t>);
    UNITTEST_LOG(result);
    UNITTEST_EQ(result.max_diff[0], 0.0f);
    return ark::unittest::SUCCESS;
}

ark::unittest::State test_relu_invalid() {
    {
        ark::Model m;
        ark::Tensor t = m.tensor({4, 2, 1024}, ark::BF16);
        ark::Tensor out = m.tensor({4, 2, 1024}, ark::FP32);
        UNITTEST_THROW(m.relu(t, out), ark::InvalidUsageError);
    }
    {
        ark::Model m;
        ark::Tensor t = m.tensor({4, 2, 1024}, ark::BF16);
        ark::Tensor out = m.tensor({4, 4, 1024}, ark::BF16);
        UNITTEST_THROW(m.relu(t, out), ark::InvalidUsageError);
    }
    return ark::unittest::SUCCESS;
}

ark::unittest::State test_math_rsqrt_fp32() {
    ark::Model m;
    ark::Tensor t = m.tensor({4, 2, 1024}, ark::FP32);
    ark::Tensor out = m.rsqrt(t);

    auto result =
        ark::op_test("math_rsqrt_fp32", m, {t}, {out}, baseline_rsqrt<float>);
    UNITTEST_LOG(result);
    UNITTEST_TRUE(result.max_diff[0] < 1e-4f);
    return ark::unittest::SUCCESS;
}

ark::unittest::State test_math_rsqrt_fp16() {
    ark::Model m;
    ark::Tensor t = m.tensor({1, 64, 1}, ark::FP16);
    ark::Tensor out = m.rsqrt(t);

    std::vector<ark::half_t> data(64, 4);

    auto result = ark::op_test("math_rsqrt_fp16", m, {t}, {out},
                               baseline_rsqrt<ark::half_t>, {data.data()});
    UNITTEST_LOG(result);
    UNITTEST_TRUE(result.max_diff[0] < 1e-4f);
    return ark::unittest::SUCCESS;
}

ark::unittest::State test_sigmoid_fp32() {
    ark::Model m;
    ark::Tensor t = m.tensor({4, 2, 1024}, ark::FP32);
    ark::Tensor out = m.sigmoid(t);

    auto result =
        ark::op_test("sigmoid_fp32", m, {t}, {out}, baseline_sigmoid<float>);
    UNITTEST_LOG(result);
    UNITTEST_TRUE(result.max_diff[0] < 1e-5f);
    return ark::unittest::SUCCESS;
}

ark::unittest::State test_sigmoid_bf16() {
    ark::Model m;
    ark::Tensor t = m.tensor({4, 2, 1024}, ark::BF16);
    ark::Tensor out = m.sigmoid(t);

    auto result = ark::op_test("sigmoid_bf16", m, {t}, {out},
                               baseline_sigmoid<ark::bfloat16_t>);
    UNITTEST_LOG(result);
    UNITTEST_TRUE(result.max_diff[0] < 1e-2f);
    return ark::unittest::SUCCESS;
}

ark::unittest::State test_sigmoid_invalid() {
    {
        ark::Model m;
        ark::Tensor t = m.tensor({4, 2, 1024}, ark::BF16);
        ark::Tensor out = m.tensor({4, 2, 1024}, ark::FP32);
        UNITTEST_THROW(m.sigmoid(t, out), ark::InvalidUsageError);
    }
    {
        ark::Model m;
        ark::Tensor t = m.tensor({4, 2, 1024}, ark::BF16);
        ark::Tensor out = m.tensor({4, 4, 1024}, ark::BF16);
        UNITTEST_THROW(m.sigmoid(t, out), ark::InvalidUsageError);
    }
    return ark::unittest::SUCCESS;
}

ark::unittest::State test_math_sqrt_fp32() {
    ark::Model m;
    ark::Tensor t = m.tensor({4, 2, 1024}, ark::FP32);
    ark::Tensor out = m.sqrt(t);

    auto result =
        ark::op_test("math_sqrt_fp32", m, {t}, {out}, baseline_sqrt<float>);
    UNITTEST_LOG(result);
    UNITTEST_TRUE(result.max_diff[0] < 1e-6f);
    return ark::unittest::SUCCESS;
}

ark::unittest::State test_math_sqrt_fp16_small_last_dim() {
    ark::Model m;
    ark::Tensor t = m.tensor({4, 1024, 1}, ark::FP16, {4, 1024, 2});
    ark::Tensor out = m.sqrt(t);

    auto result = ark::op_test("math_sqrt_fp16_small_last_dim", m, {t}, {out},
                               baseline_sqrt<ark::half_t>);
    UNITTEST_LOG(result);
    UNITTEST_TRUE(result.max_diff[0] < 1e-4f);
    return ark::unittest::SUCCESS;
}

ark::unittest::State test_math_sqrt_invalid() {
    {
        ark::Model model;
        ark::Tensor input = model.tensor({1, 3, 16, 8192}, ark::FP32);
        ark::Tensor output = model.tensor({1, 3, 16, 8192}, ark::FP16);
        UNITTEST_THROW(model.sqrt(input, output), ark::InvalidUsageError);
    }
    {
        ark::Model model;
        ark::Tensor input = model.tensor({1, 3, 16, 8192}, ark::FP32);
        ark::Tensor output = model.tensor({1, 3, 16, 1024}, ark::FP32);
        UNITTEST_THROW(model.sqrt(input, output), ark::InvalidUsageError);
    }
    return ark::unittest::SUCCESS;
}

int main() {
    ark::init();
    UNITTEST(test_gelu_fp32);
    UNITTEST(test_gelu_bf16);
    UNITTEST(test_gelu_invalid);
    UNITTEST(test_exp_fp32);
    UNITTEST(test_exp_fp16);
    UNITTEST(test_exp_invalid);
    UNITTEST(test_relu_fp32);
    UNITTEST(test_relu_fp16);
    UNITTEST(test_relu_bf16);
    UNITTEST(test_relu_invalid);
    UNITTEST(test_math_rsqrt_fp32);
    UNITTEST(test_math_rsqrt_fp16);
    UNITTEST(test_sigmoid_fp32);
    UNITTEST(test_sigmoid_bf16);
    UNITTEST(test_sigmoid_invalid);
    UNITTEST(test_math_sqrt_fp32);
    UNITTEST(test_math_sqrt_fp16_small_last_dim);
    UNITTEST(test_math_sqrt_invalid);
    return ark::unittest::SUCCESS;
}
