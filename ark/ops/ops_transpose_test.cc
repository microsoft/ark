// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "include/ark.h"
#include "include/ark_utils.h"
#include "ops_test_common.h"
#include "unittest/unittest_utils.h"

template <typename T>
void baseline_transpose_0132(std::vector<void *> &outputs,
                             const std::vector<ark::Dims> &output_shapes,
                             const std::vector<void *> &inputs,
                             const std::vector<ark::Dims> &input_shapes) {
    T *out = static_cast<T *>(outputs[0]);
    T *in = static_cast<T *>(inputs[0]);
    ark::Dims osh = output_shapes[0].dims4();
    ark::Dims ish = input_shapes[0].dims4();
    for (ark::DimType n = 0; n < ish[0]; ++n) {
        for (ark::DimType c = 0; c < ish[1]; ++c) {
            for (ark::DimType h = 0; h < ish[2]; ++h) {
                for (ark::DimType w = 0; w < ish[3]; ++w) {
                    // out[n][c][w][h] = in[n][c][h][w]
                    out[h + w * osh[3] + c * osh[2] * osh[3] +
                        n * osh[1] * osh[2] * osh[3]] =
                        in[w + h * ish[3] + c * ish[3] * ish[2] +
                           n * ish[3] * ish[2] * ish[1]];
                }
            }
        }
    }
};

template <typename T>
void baseline_transpose_0231(std::vector<void *> &outputs,
                             const std::vector<ark::Dims> &output_shapes,
                             const std::vector<void *> &inputs,
                             const std::vector<ark::Dims> &input_shapes) {
    T *out = static_cast<T *>(outputs[0]);
    T *in = static_cast<T *>(inputs[0]);
    ark::Dims osh = output_shapes[0].dims4();
    ark::Dims ish = input_shapes[0].dims4();
    for (ark::DimType n = 0; n < ish[0]; ++n) {
        for (ark::DimType c = 0; c < ish[1]; ++c) {
            for (ark::DimType h = 0; h < ish[2]; ++h) {
                for (ark::DimType w = 0; w < ish[3]; ++w) {
                    // out[n][h][w][c] = in[n][c][h][w]
                    out[c + w * osh[3] + h * osh[2] * osh[3] +
                        n * osh[1] * osh[2] * osh[3]] =
                        in[w + h * ish[3] + c * ish[3] * ish[2] +
                           n * ish[3] * ish[2] * ish[1]];
                }
            }
        }
    }
};

ark::unittest::State test_transpose_0132_fp32() {
    ark::Model m;
    ark::Tensor *t = m.tensor({5, 3, 32, 128}, ark::FP32);
    ark::Tensor *out = m.transpose(t, {0, 1, 3, 2});

    auto result = ark::op_test("transpose_0132_fp32", m, {t}, {out},
                               baseline_transpose_0132<float>);
    UNITTEST_LOG(result);
    UNITTEST_EQ(result.max_diff[0], 0.0f);
    return ark::unittest::SUCCESS;
}

ark::unittest::State test_transpose_0132_fp16() {
    ark::Model m;
    ark::Tensor *t = m.tensor({5, 3, 32, 128}, ark::FP16);
    ark::Tensor *out = m.transpose(t, {0, 1, 3, 2});

    auto result = ark::op_test("transpose_0132_fp32", m, {t}, {out},
                               baseline_transpose_0132<ark::half_t>);
    UNITTEST_LOG(result);
    UNITTEST_EQ(result.max_diff[0], 0.0f);
    return ark::unittest::SUCCESS;
}

ark::unittest::State test_transpose_0231_fp32() {
    ark::Model m;
    ark::Tensor *t = m.tensor({5, 3, 32, 128}, ark::FP32);
    ark::Tensor *out = m.transpose(t, {0, 2, 3, 1});

    auto result = ark::op_test("transpose_0231_fp32", m, {t}, {out},
                               baseline_transpose_0231<float>);
    UNITTEST_LOG(result);
    UNITTEST_EQ(result.max_diff[0], 0.0f);
    return ark::unittest::SUCCESS;
}

ark::unittest::State test_transpose_0231_fp16() {
    ark::Model m;
    ark::Tensor *t = m.tensor({5, 3, 32, 128}, ark::FP16);
    ark::Tensor *out = m.transpose(t, {0, 2, 3, 1});

    auto result = ark::op_test("transpose_0231_fp16", m, {t}, {out},
                               baseline_transpose_0231<ark::half_t>);
    UNITTEST_LOG(result);
    UNITTEST_EQ(result.max_diff[0], 0.0f);
    return ark::unittest::SUCCESS;
}

int main() {
    ark::init();
    UNITTEST(test_transpose_0132_fp32);
    UNITTEST(test_transpose_0132_fp16);
    UNITTEST(test_transpose_0231_fp32);
    UNITTEST(test_transpose_0231_fp16);
    return ark::unittest::SUCCESS;
}
