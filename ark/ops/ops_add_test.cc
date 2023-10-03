// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "include/ark.h"
#include "include/ark_utils.h"
#include "ops_test_common.h"
#include "unittest/unittest_utils.h"

template <typename T>
void baseline_add(std::vector<void *> &outputs,
                  const std::vector<ark::Dims> &output_shapes,
                  const std::vector<void *> &inputs,
                  const std::vector<ark::Dims> &input_shapes) {
    T *out = static_cast<T *>(outputs[0]);
    T *t0 = static_cast<T *>(inputs[0]);
    T *t1 = static_cast<T *>(inputs[1]);

    // NumPy-style broadcasted addition
    ark::Dims osh = output_shapes[0].dims4();
    ark::Dims ish0 = input_shapes[0].dims4();
    ark::Dims ish1 = input_shapes[1].dims4();
    for (ark::DimType n = 0; n < osh[0]; ++n) {
        for (ark::DimType c = 0; c < osh[1]; ++c) {
            for (ark::DimType h = 0; h < osh[2]; ++h) {
                for (ark::DimType w = 0; w < osh[3]; ++w) {
                    out[w + h * osh[3] + c * osh[2] * osh[3] +
                        n * osh[1] * osh[2] * osh[3]] =
                        t0[(w % ish0[3]) + (h % ish0[2]) * ish0[3] +
                           (c % ish0[1]) * ish0[2] * ish0[3] +
                           (n % ish0[0]) * ish0[1] * ish0[2] * ish0[3]] +
                        t1[(w % ish1[3]) + (h % ish1[2]) * ish1[3] +
                           (c % ish1[1]) * ish1[2] * ish1[3] +
                           (n % ish1[0]) * ish1[1] * ish1[2] * ish1[3]];
                }
            }
        }
    }
};

ark::unittest::State test_add_fp32() {
    ark::Model m;
    ark::Tensor *t0 = m.tensor(ark::Dims(8192), ark::FP32);
    ark::Tensor *t1 = m.tensor(ark::Dims(8192), ark::FP32);
    ark::Tensor *out = m.add(t0, t1);

    auto result =
        ark::op_test("add_fp32", m, {t0, t1}, {out}, baseline_add<float>);
    UNITTEST_LOG(result);
    UNITTEST_EQ(result.max_diff[0], 0.0f);
    return ark::unittest::SUCCESS;
}

ark::unittest::State test_add_fp16() {
    ark::Model m;
    ark::Tensor *t0 = m.tensor(ark::Dims(8192), ark::FP16);
    ark::Tensor *t1 = m.tensor(ark::Dims(8192), ark::FP16);
    ark::Tensor *out = m.add(t0, t1);

    auto result =
        ark::op_test("add_fp16", m, {t0, t1}, {out}, baseline_add<ark::half_t>);
    UNITTEST_LOG(result);
    UNITTEST_EQ(result.max_diff[0], 0.0f);
    return ark::unittest::SUCCESS;
}

ark::unittest::State test_add_overwrite() {
    ark::Model m;
    ark::Tensor *t0 = m.tensor(ark::Dims(8192), ark::FP16);
    ark::Tensor *t1 = m.tensor(ark::Dims(8192), ark::FP16);
    ark::Tensor *out = m.add(t0, t1, t1);

    auto result = ark::op_test("add_overwrite", m, {t0, t1}, {out},
                               baseline_add<ark::half_t>);
    UNITTEST_LOG(result);
    UNITTEST_EQ(result.max_diff[0], 0.0f);
    return ark::unittest::SUCCESS;
}

ark::unittest::State test_add_broadcast() {
    {
        ark::Model m;
        ark::Tensor *t0 = m.tensor(ark::Dims(4, 1024), ark::FP16);
        ark::Tensor *t1 = m.tensor(ark::Dims(1, 1024), ark::FP16);
        ark::Tensor *out = m.add(t0, t1);

        auto result = ark::op_test("add_broadcast", m, {t0, t1}, {out},
                                   baseline_add<ark::half_t>);
        UNITTEST_LOG(result);
        UNITTEST_EQ(result.max_diff[0], 0.0f);
    }
    {
        ark::Model m;
        ark::Tensor *t0 = m.tensor(ark::Dims(4, 1024), ark::FP16);
        ark::Tensor *t1 = m.tensor(ark::Dims(4, 1), ark::FP16);
        ark::Tensor *out = m.add(t0, t1);

        auto result = ark::op_test("add_broadcast", m, {t0, t1}, {out},
                                   baseline_add<ark::half_t>);
        UNITTEST_LOG(result);
        UNITTEST_EQ(result.max_diff[0], 0.0f);
    }
    {
        ark::Model m;
        ark::Tensor *t0 = m.tensor(ark::Dims(3, 1, 1024), ark::FP16);
        ark::Tensor *t1 = m.tensor(ark::Dims(1, 4, 1), ark::FP16);
        ark::Tensor *out = m.add(t0, t1);

        auto result = ark::op_test("add_broadcast", m, {t0, t1}, {out},
                                   baseline_add<ark::half_t>);
        UNITTEST_LOG(result);
        UNITTEST_EQ(result.max_diff[0], 0.0f);
    }
    return ark::unittest::SUCCESS;
}

int main() {
    ark::init();
    UNITTEST(test_add_fp32);
    UNITTEST(test_add_fp16);
    UNITTEST(test_add_overwrite);
    UNITTEST(test_add_broadcast);
    return ark::unittest::SUCCESS;
}
