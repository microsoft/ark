// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include <cassert>
#include <numeric>

#include "include/ark.h"
#include "ops_test_common.h"
#include "unittest/unittest_utils.h"

template <typename T>
void baseline_reduce_sum_axis0(std::vector<void *> &outputs,
                               const std::vector<ark::Dims> &output_shapes,
                               const std::vector<void *> &inputs,
                               const std::vector<ark::Dims> &input_shapes,
                               int) {
    T *out = static_cast<T *>(outputs[0]);
    T *input = static_cast<T *>(inputs[0]);

    ark::Dims osh = output_shapes[0].dims4();
    ark::Dims ish = input_shapes[0].dims4();

    assert(osh[0] == 1);

    for (ark::DimType c = 0; c < ish[1]; ++c) {
        for (ark::DimType h = 0; h < ish[2]; ++h) {
            for (ark::DimType w = 0; w < ish[3]; ++w) {
                float sum = 0;
                for (ark::DimType n = 0; n < ish[0]; ++n) {
                    sum += float(input[n * ish[1] * ish[2] * ish[3] +
                                       c * ish[2] * ish[3] + h * ish[3] + w]);
                }
                out[c * osh[2] * osh[3] + h * osh[3] + w] = T(sum);
            }
        }
    }
}

template <typename T>
void baseline_reduce_sum_axis1(std::vector<void *> &outputs,
                               const std::vector<ark::Dims> &output_shapes,
                               const std::vector<void *> &inputs,
                               const std::vector<ark::Dims> &input_shapes,
                               int) {
    T *out = static_cast<T *>(outputs[0]);
    T *input = static_cast<T *>(inputs[0]);

    ark::Dims osh = output_shapes[0].dims4();
    ark::Dims ish = input_shapes[0].dims4();

    assert(osh[1] == 1);

    for (ark::DimType n = 0; n < ish[0]; ++n) {
        for (ark::DimType h = 0; h < ish[2]; ++h) {
            for (ark::DimType w = 0; w < ish[3]; ++w) {
                float sum = 0;
                for (ark::DimType c = 0; c < ish[1]; ++c) {
                    sum += float(input[n * ish[1] * ish[2] * ish[3] +
                                       c * ish[2] * ish[3] + h * ish[3] + w]);
                }
                out[n * osh[1] * osh[2] * osh[3] + h * osh[3] + w] = T(sum);
            }
        }
    }
}

template <typename T>
void baseline_reduce_sum_axis2(std::vector<void *> &outputs,
                               const std::vector<ark::Dims> &output_shapes,
                               const std::vector<void *> &inputs,
                               const std::vector<ark::Dims> &input_shapes,
                               int) {
    T *out = static_cast<T *>(outputs[0]);
    T *input = static_cast<T *>(inputs[0]);

    ark::Dims osh = output_shapes[0].dims4();
    ark::Dims ish = input_shapes[0].dims4();

    assert(osh[2] == 1);

    for (ark::DimType n = 0; n < ish[0]; ++n) {
        for (ark::DimType c = 0; c < ish[1]; ++c) {
            for (ark::DimType w = 0; w < ish[3]; ++w) {
                float sum = 0;
                for (ark::DimType h = 0; h < ish[2]; ++h) {
                    sum += float(input[n * ish[1] * ish[2] * ish[3] +
                                       c * ish[2] * ish[3] + h * ish[3] + w]);
                }
                out[n * osh[1] * osh[2] * osh[3] + c * osh[2] * osh[3] + w] =
                    T(sum);
            }
        }
    }
};

template <typename T>
void baseline_reduce_sum_axis3(std::vector<void *> &outputs,
                               const std::vector<ark::Dims> &output_shapes,
                               const std::vector<void *> &inputs,
                               const std::vector<ark::Dims> &input_shapes,
                               int) {
    T *out = static_cast<T *>(outputs[0]);
    T *input = static_cast<T *>(inputs[0]);

    ark::Dims osh = output_shapes[0].dims4();
    ark::Dims ish = input_shapes[0].dims4();

    assert(osh[3] == 1);

    for (ark::DimType n = 0; n < ish[0]; ++n) {
        for (ark::DimType c = 0; c < ish[1]; ++c) {
            for (ark::DimType h = 0; h < ish[2]; ++h) {
                float sum = 0;
                for (ark::DimType w = 0; w < ish[3]; ++w) {
                    sum += float(input[n * ish[1] * ish[2] * ish[3] +
                                       c * ish[2] * ish[3] + h * ish[3] + w]);
                }
                out[n * osh[1] * osh[2] * osh[3] + c * osh[2] * osh[3] +
                    h * osh[3]] = T(sum);
            }
        }
    }
};

ark::unittest::State test_reduce_axis0() {
    ark::Model m;
    ark::Tensor *t = m.tensor(ark::Dims(7, 2, 4, 1024), ark::FP32);
    ark::Tensor *out = m.reduce_sum(t, /*axis=*/0);

    auto result = ark::op_test("reduce_axis0", m, {t}, {out},
                               baseline_reduce_sum_axis0<float>);
    UNITTEST_LOG(result);
    UNITTEST_EQ(result.max_diff[0], 0.0f);
    return ark::unittest::SUCCESS;
}

ark::unittest::State test_reduce_axis1() {
    ark::Model m;
    ark::Tensor *t = m.tensor(ark::Dims(1, 2, 4, 1024), ark::FP32);
    ark::Tensor *out = m.reduce_sum(t, /*axis=*/1);

    auto result = ark::op_test("reduce_axis1", m, {t}, {out},
                               baseline_reduce_sum_axis1<float>);
    UNITTEST_LOG(result);
    UNITTEST_EQ(result.max_diff[0], 0.0f);
    return ark::unittest::SUCCESS;
}

ark::unittest::State test_reduce_axis2() {
    ark::Model m;
    ark::Tensor *t = m.tensor(ark::Dims(1, 1, 7, 8192), ark::FP32);
    ark::Tensor *out = m.reduce_sum(t, /*axis=*/2);

    auto result = ark::op_test("reduce_axis2", m, {t}, {out},
                               baseline_reduce_sum_axis2<float>);
    UNITTEST_LOG(result);
    UNITTEST_EQ(result.max_diff[0], 0.0f);
    return ark::unittest::SUCCESS;
}

ark::unittest::State test_reduce_axis3() {
    ark::Model m;
    ark::Tensor *t = m.tensor(ark::Dims(1, 1, 2, 8192), ark::FP32);
    ark::Tensor *out = m.reduce_sum(t, /*axis=*/3);

    auto result = ark::op_test("reduce_axis3", m, {t}, {out},
                               baseline_reduce_sum_axis3<float>);
    UNITTEST_LOG(result);
    UNITTEST_TRUE(result.max_err_rate[0] < 1e-4f);
    return ark::unittest::SUCCESS;
}

ark::unittest::State test_reduce_axis3_padded() {
    ark::Model m;
    ark::Tensor *t = m.tensor(ark::Dims(1, 1, 2, 8192), ark::FP32);
    ark::Tensor *out = m.tensor(ark::Dims(1, 1, 2, 1), ark::FP32, nullptr,
                                ark::Dims(1, 1, 2, 32));
    out = m.reduce_sum(t, /*axis=*/3, out);

    auto result = ark::op_test("reduce_axis3_padded", m, {t}, {out},
                               baseline_reduce_sum_axis3<float>);
    UNITTEST_LOG(result);
    UNITTEST_TRUE(result.max_err_rate[0] < 1e-4f);
    return ark::unittest::SUCCESS;
}

ark::unittest::State test_reduce_fp16() {
    {
        ark::Model m;
        ark::Tensor *t = m.tensor(ark::Dims(7, 2, 4, 1024), ark::FP16);
        ark::Tensor *out = m.reduce_sum(t, /*axis=*/0);

        auto result = ark::op_test("reduce_fp16_axis0", m, {t}, {out},
                                   baseline_reduce_sum_axis0<ark::half_t>);
        UNITTEST_LOG(result);
        UNITTEST_TRUE(result.max_diff[0] < 1e-2f);
    }
    {
        ark::Model m;
        ark::Tensor *t = m.tensor(ark::Dims(7, 2, 4, 1024), ark::FP16);
        ark::Tensor *out = m.reduce_sum(t, /*axis=*/3);

        auto result = ark::op_test("reduce_fp16_axis3", m, {t}, {out},
                                   baseline_reduce_sum_axis3<ark::half_t>);
        UNITTEST_LOG(result);
    }
    return ark::unittest::SUCCESS;
}

ark::unittest::State test_reduce_bf16() {
    std::vector<ark::bfloat16_t> data_vec(7 * 2 * 4 * 1024);
    for (size_t i = 0; i < data_vec.size(); ++i) {
        data_vec[i] = ark::bfloat16_t((i % 1000) * 1e-4f);
    }

    {
        ark::Model m;
        ark::Tensor *t = m.tensor(ark::Dims(7, 2, 4, 1024), ark::BF16);
        ark::Tensor *out = m.reduce_sum(t, /*axis=*/0);

        auto result = ark::op_test("reduce_bf16_axis0", m, {t}, {out},
                                   baseline_reduce_sum_axis0<ark::bfloat16_t>,
                                   {data_vec.data()});
        UNITTEST_LOG(result);
        UNITTEST_TRUE(result.max_diff[0] < 1e-2f);
    }
    {
        ark::Model m;
        ark::Tensor *t = m.tensor(ark::Dims(7, 2, 4, 1024), ark::BF16);
        ark::Tensor *out = m.reduce_sum(t, /*axis=*/3);

        auto result = ark::op_test("reduce_bf16_axis3", m, {t}, {out},
                                   baseline_reduce_sum_axis3<ark::bfloat16_t>,
                                   {data_vec.data()});
        UNITTEST_LOG(result);
    }
    return ark::unittest::SUCCESS;
}

int main() {
    ark::init();
    UNITTEST(test_reduce_axis0);
    UNITTEST(test_reduce_axis1);
    UNITTEST(test_reduce_axis2);
    UNITTEST(test_reduce_axis3);
    UNITTEST(test_reduce_axis3_padded);
    UNITTEST(test_reduce_fp16);
    UNITTEST(test_reduce_bf16);
    return ark::unittest::SUCCESS;
}
