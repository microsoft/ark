// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include <cassert>
#include <limits>
#include <numeric>

#include "ops_test_common.hpp"

template <typename T, bool KeepDim = true>
void baseline_reduce_sum_axis0(std::vector<void *> &outputs,
                               const std::vector<ark::Dims> &output_shapes,
                               const std::vector<void *> &inputs,
                               const std::vector<ark::Dims> &input_shapes,
                               int) {
    T *out = static_cast<T *>(outputs[0]);
    T *input = static_cast<T *>(inputs[0]);

    ark::Dims osh = output_shapes[0];
    ark::Dims ish = input_shapes[0].dims4();

    if (KeepDim) {
        assert(osh[0] == 1);
    } else {
        osh.insert(0, 1);
    }
    osh = osh.dims4();

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

template <typename T, bool KeepDim = true>
void baseline_reduce_sum_axis1(std::vector<void *> &outputs,
                               const std::vector<ark::Dims> &output_shapes,
                               const std::vector<void *> &inputs,
                               const std::vector<ark::Dims> &input_shapes,
                               int) {
    T *out = static_cast<T *>(outputs[0]);
    T *input = static_cast<T *>(inputs[0]);

    ark::Dims osh = output_shapes[0];
    ark::Dims ish = input_shapes[0].dims4();

    if (KeepDim) {
        assert(osh[1] == 1);
    } else {
        osh.insert(1, 1);
    }
    osh = osh.dims4();

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

template <typename T, bool KeepDim = true>
void baseline_reduce_sum_axis2(std::vector<void *> &outputs,
                               const std::vector<ark::Dims> &output_shapes,
                               const std::vector<void *> &inputs,
                               const std::vector<ark::Dims> &input_shapes,
                               int) {
    T *out = static_cast<T *>(outputs[0]);
    T *input = static_cast<T *>(inputs[0]);

    ark::Dims osh = output_shapes[0];
    ark::Dims ish = input_shapes[0].dims4();

    if (KeepDim) {
        assert(osh[2] == 1);
    } else {
        osh.insert(2, 1);
    }
    osh = osh.dims4();

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

template <typename T, bool KeepDim = true>
void baseline_reduce_sum_axis3(std::vector<void *> &outputs,
                               const std::vector<ark::Dims> &output_shapes,
                               const std::vector<void *> &inputs,
                               const std::vector<ark::Dims> &input_shapes,
                               int) {
    T *out = static_cast<T *>(outputs[0]);
    T *input = static_cast<T *>(inputs[0]);

    ark::Dims osh = output_shapes[0];
    ark::Dims ish = input_shapes[0].dims4();

    if (KeepDim) {
        assert(osh[3] == 1);
    } else {
        osh.insert(3, 1);
    }
    osh = osh.dims4();

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

template <typename T, bool KeepDim = true>
void baseline_reduce_max_axis3(std::vector<void *> &outputs,
                               const std::vector<ark::Dims> &output_shapes,
                               const std::vector<void *> &inputs,
                               const std::vector<ark::Dims> &input_shapes,
                               int) {
    T *out = static_cast<T *>(outputs[0]);
    T *input = static_cast<T *>(inputs[0]);

    ark::Dims osh = output_shapes[0];
    ark::Dims ish = input_shapes[0].dims4();

    if (KeepDim) {
        assert(osh[3] == 1);
    } else {
        osh.insert(3, 1);
    }
    osh = osh.dims4();

    for (ark::DimType n = 0; n < ish[0]; ++n) {
        for (ark::DimType c = 0; c < ish[1]; ++c) {
            for (ark::DimType h = 0; h < ish[2]; ++h) {
                float max_val = std::numeric_limits<float>::lowest();
                for (ark::DimType w = 0; w < ish[3]; ++w) {
                    float val =
                        float(input[n * ish[1] * ish[2] * ish[3] +
                                    c * ish[2] * ish[3] + h * ish[3] + w]);
                    max_val = std::max(max_val, val);
                }
                out[n * osh[1] * osh[2] * osh[3] + c * osh[2] * osh[3] +
                    h * osh[3]] = T(max_val);
            }
        }
    }
};

template <typename T, bool KeepDim = true>
void baseline_reduce_mean_axis3(std::vector<void *> &outputs,
                                const std::vector<ark::Dims> &output_shapes,
                                const std::vector<void *> &inputs,
                                const std::vector<ark::Dims> &input_shapes,
                                int) {
    T *out = static_cast<T *>(outputs[0]);
    T *input = static_cast<T *>(inputs[0]);

    ark::Dims osh = output_shapes[0];
    ark::Dims ish = input_shapes[0].dims4();

    if (KeepDim) {
        assert(osh[3] == 1);
    } else {
        osh.insert(3, 1);
    }
    osh = osh.dims4();

    for (ark::DimType n = 0; n < ish[0]; ++n) {
        for (ark::DimType c = 0; c < ish[1]; ++c) {
            for (ark::DimType h = 0; h < ish[2]; ++h) {
                float mean = 0;
                for (ark::DimType w = 0; w < ish[3]; ++w) {
                    mean += float(input[n * ish[1] * ish[2] * ish[3] +
                                        c * ish[2] * ish[3] + h * ish[3] + w]);
                }
                mean /= ish[3];
                out[n * osh[1] * osh[2] * osh[3] + c * osh[2] * osh[3] +
                    h * osh[3]] = T(mean);
            }
        }
    }
};

ark::unittest::State test_reduce_sum_axis0() {
    ark::Model m;
    ark::Tensor t = m.tensor(ark::Dims(7, 2, 4, 1024), ark::FP32);
    ark::Tensor out = m.reduce_sum(t, /*axis=*/0);

    auto result = ark::op_test("reduce_sum_axis0", m, {t}, {out},
                               baseline_reduce_sum_axis0<float>);
    UNITTEST_LOG(result);
    UNITTEST_TRUE(result.max_diff[0] <
                  ark::reduction_abs_error_bound<float>(0.1, t.shape()[0]));
    return ark::unittest::SUCCESS;
}

ark::unittest::State test_reduce_sum_axis1() {
    ark::Model m;
    ark::Tensor t = m.tensor(ark::Dims(1, 2, 4, 1024), ark::FP32);
    ark::Tensor out = m.reduce_sum(t, /*axis=*/1);

    auto result = ark::op_test("reduce_sum_axis1", m, {t}, {out},
                               baseline_reduce_sum_axis1<float>);
    UNITTEST_LOG(result);
    UNITTEST_TRUE(result.max_diff[0] <
                  ark::reduction_abs_error_bound<float>(0.1, t.shape()[1]));
    return ark::unittest::SUCCESS;
}

ark::unittest::State test_reduce_sum_axis2() {
    ark::Model m;
    ark::Tensor t = m.tensor(ark::Dims(1, 1, 7, 8192), ark::FP32);
    ark::Tensor out = m.reduce_sum(t, /*axis=*/2);

    auto result = ark::op_test("reduce_sum_axis2", m, {t}, {out},
                               baseline_reduce_sum_axis2<float>);
    UNITTEST_LOG(result);
    UNITTEST_TRUE(result.max_diff[0] <
                  ark::reduction_abs_error_bound<float>(0.1, t.shape()[2]));
    return ark::unittest::SUCCESS;
}

ark::unittest::State test_reduce_sum_axis3() {
    ark::Model m;
    ark::Tensor t = m.tensor(ark::Dims(1, 1, 2, 8192), ark::FP32);
    ark::Tensor out = m.reduce_sum(t, /*axis=*/3);

    auto result = ark::op_test("reduce_sum_axis3", m, {t}, {out},
                               baseline_reduce_sum_axis3<float>);
    UNITTEST_LOG(result);
    UNITTEST_TRUE(result.max_diff[0] <
                  ark::reduction_abs_error_bound<float>(0.1, t.shape()[3]));
    return ark::unittest::SUCCESS;
}

ark::unittest::State test_reduce_sum_axis3_padded() {
    ark::Model m;
    ark::Tensor t = m.tensor(ark::Dims(1, 1, 2, 8192), ark::FP32);
    ark::Tensor out =
        m.tensor(ark::Dims(1, 1, 2, 1), ark::FP32, ark::Dims(1, 1, 2, 32));
    out = m.reduce_sum(t, /*axis=*/3, true, out);

    auto result = ark::op_test("reduce_sum_axis3_padded", m, {t}, {out},
                               baseline_reduce_sum_axis3<float>);
    UNITTEST_LOG(result);
    UNITTEST_TRUE(result.max_diff[0] <
                  ark::reduction_abs_error_bound<float>(0.1, t.shape()[3]));
    return ark::unittest::SUCCESS;
}

ark::unittest::State test_reduce_sum_fp16() {
    {
        ark::Model m;
        ark::Tensor t = m.tensor(ark::Dims(7, 2, 4, 1024), ark::FP16);
        ark::Tensor out = m.reduce_sum(t, /*axis=*/0);

        auto result = ark::op_test("reduce_sum_fp16_axis0", m, {t}, {out},
                                   baseline_reduce_sum_axis0<ark::half_t>);
        UNITTEST_LOG(result);
        UNITTEST_TRUE(
            result.max_diff[0] <
            ark::reduction_abs_error_bound<ark::half_t>(0.1, t.shape()[0]));
    }
    {
        ark::Model m;
        ark::Tensor t = m.tensor(ark::Dims(7, 2, 4, 1024), ark::FP16);
        ark::Tensor out = m.reduce_sum(t, /*axis=*/3);

        auto result = ark::op_test("reduce_sum_fp16_axis3", m, {t}, {out},
                                   baseline_reduce_sum_axis3<ark::half_t>);
        UNITTEST_LOG(result);
        UNITTEST_TRUE(
            result.max_diff[0] <
            ark::reduction_abs_error_bound<ark::half_t>(0.1, t.shape()[3]));
    }
    return ark::unittest::SUCCESS;
}

ark::unittest::State test_reduce_sum_bf16() {
    std::vector<ark::bf16> data_vec(7 * 2 * 4 * 256);
    for (size_t i = 0; i < data_vec.size(); ++i) {
        data_vec[i] = ark::bf16((i % 1000) * 1e-4f);
    }

    {
        ark::Model m;
        ark::Tensor t = m.tensor(ark::Dims(7, 2, 4, 256), ark::BF16);
        ark::Tensor out = m.reduce_sum(t, /*axis=*/0);

        auto result = ark::op_test("reduce_sum_bf16_axis0", m, {t}, {out},
                                   baseline_reduce_sum_axis0<ark::bf16>,
                                   {data_vec.data()});
        UNITTEST_LOG(result);
        UNITTEST_TRUE(
            result.max_diff[0] <
            ark::reduction_abs_error_bound<ark::bf16>(0.1, t.shape()[0]));
    }
    {
        ark::Model m;
        ark::Tensor t = m.tensor(ark::Dims(7, 2, 4, 256), ark::BF16);
        ark::Tensor out = m.reduce_sum(t, /*axis=*/3);

        auto result = ark::op_test("reduce_sum_bf16_axis3", m, {t}, {out},
                                   baseline_reduce_sum_axis3<ark::bf16>,
                                   {data_vec.data()});
        UNITTEST_LOG(result);
        UNITTEST_TRUE(
            result.max_diff[0] <
            ark::reduction_abs_error_bound<ark::bf16>(0.1, t.shape()[3]));
    }
    return ark::unittest::SUCCESS;
}

ark::unittest::State test_reduce_sum_fp16_no_keepdims() {
    {
        ark::Model m;
        ark::Tensor t = m.tensor(ark::Dims(7, 2, 4, 1024), ark::FP16);
        ark::Tensor out = m.reduce_sum(t, /*axis=*/0, false);

        UNITTEST_EQ(out.shape(), ark::Dims(2, 4, 1024));

        auto result =
            ark::op_test("reduce_sum_fp16_axis0", m, {t}, {out},
                         baseline_reduce_sum_axis0<ark::half_t, false>);
        UNITTEST_LOG(result);
        UNITTEST_TRUE(
            result.max_diff[0] <
            ark::reduction_abs_error_bound<ark::half_t>(0.1, t.shape()[0]));
    }
    {
        ark::Model m;
        ark::Tensor t = m.tensor(ark::Dims(7, 2, 4, 1024), ark::FP16);
        ark::Tensor out = m.reduce_sum(t, /*axis=*/3, false);

        UNITTEST_EQ(out.shape(), ark::Dims(7, 2, 4));

        auto result =
            ark::op_test("reduce_sum_fp16_axis3", m, {t}, {out},
                         baseline_reduce_sum_axis3<ark::half_t, false>);
        UNITTEST_LOG(result);
        UNITTEST_TRUE(
            result.max_diff[0] <
            ark::reduction_abs_error_bound<ark::half_t>(0.1, t.shape()[3]));
    }
    return ark::unittest::SUCCESS;
}

ark::unittest::State test_reduce_sum_invalid() {
    {
        ark::Model m;
        ark::Tensor t = m.tensor(ark::Dims(7, 2, 4, 1024), ark::BF16);
        ark::Tensor out = m.tensor(ark::Dims(1, 2, 4, 1024), ark::FP32);
        UNITTEST_THROW(m.reduce_sum(t, /*axis=*/0, true, out),
                       ark::InvalidUsageError);
    }
    {
        ark::Model m;
        ark::Tensor t = m.tensor(ark::Dims(7, 2, 4, 1024), ark::BF16);
        ark::Tensor out = m.tensor(ark::Dims(7, 2, 4, 1), ark::FP32);
        UNITTEST_THROW(m.reduce_sum(t, /*axis=*/3, true, out),
                       ark::InvalidUsageError);
    }
    {
        ark::Model m;
        ark::Tensor t = m.tensor(ark::Dims(7, 2, 4, 1024), ark::BF16);
        ark::Tensor out = m.tensor(ark::Dims(1, 2, 4, 512), ark::BF16);
        UNITTEST_THROW(m.reduce_sum(t, /*axis=*/0, true, out),
                       ark::InvalidUsageError);
    }
    {
        ark::Model m;
        ark::Tensor t = m.tensor(ark::Dims(7, 2, 4, 1024), ark::BF16);
        ark::Tensor out = m.tensor(ark::Dims(7, 1, 4, 1), ark::BF16);
        UNITTEST_THROW(m.reduce_sum(t, /*axis=*/3, true, out),
                       ark::InvalidUsageError);
    }
    {
        ark::Model m;
        ark::Tensor t = m.tensor(ark::Dims(7, 2, 4, 1024), ark::BF16);
        ark::Tensor out = m.tensor(ark::Dims(3, 2, 4, 1024), ark::BF16);
        UNITTEST_THROW(m.reduce_sum(t, /*axis=*/0, true, out),
                       ark::InvalidUsageError);
    }
    {
        ark::Model m;
        ark::Tensor t = m.tensor(ark::Dims(7, 2, 4, 1024), ark::BF16);
        ark::Tensor out = m.tensor(ark::Dims(7, 2, 4, 3), ark::BF16);
        UNITTEST_THROW(m.reduce_sum(t, /*axis=*/3, true, out),
                       ark::InvalidUsageError);
    }
    {
        ark::Model m;
        ark::Tensor t = m.tensor(ark::Dims(7, 2, 4, 1), ark::BF16);
        UNITTEST_THROW(m.reduce_sum(t, /*axis=*/3, true, t),
                       ark::InvalidUsageError);
    }
    return ark::unittest::SUCCESS;
}

ark::unittest::State test_reduce_max_axis3() {
    ark::Model m;
    ark::Tensor t = m.tensor(ark::Dims(1, 1, 2, 8192), ark::FP32);
    ark::Tensor out = m.reduce_max(t, /*axis=*/3);

    auto result = ark::op_test("reduce_max_axis3", m, {t}, {out},
                               baseline_reduce_max_axis3<float>);
    UNITTEST_LOG(result);
    UNITTEST_EQ(result.max_diff[0], 0);
    return ark::unittest::SUCCESS;
}

ark::unittest::State test_reduce_mean_axis3() {
    ark::Model m;
    ark::Tensor t = m.tensor(ark::Dims(1, 1, 2, 8192), ark::FP32);
    ark::Tensor out = m.reduce_mean(t, /*axis=*/3);

    auto result = ark::op_test("reduce_mean_axis3", m, {t}, {out},
                               baseline_reduce_mean_axis3<float>);
    UNITTEST_LOG(result);
    UNITTEST_TRUE(result.max_diff[0] <
                  ark::reduction_abs_error_bound<float>(0.1, t.shape()[3]) /
                      t.shape()[3]);
    return ark::unittest::SUCCESS;
}

int main() {
    ark::init();
    UNITTEST(test_reduce_sum_axis0);
    UNITTEST(test_reduce_sum_axis1);
    UNITTEST(test_reduce_sum_axis2);
    UNITTEST(test_reduce_sum_axis3);
    UNITTEST(test_reduce_sum_axis3_padded);
    UNITTEST(test_reduce_sum_fp16);
    UNITTEST(test_reduce_sum_bf16);
    UNITTEST(test_reduce_sum_fp16_no_keepdims);
    UNITTEST(test_reduce_sum_invalid);
    UNITTEST(test_reduce_max_axis3);
    UNITTEST(test_reduce_mean_axis3);
    return ark::unittest::SUCCESS;
}
