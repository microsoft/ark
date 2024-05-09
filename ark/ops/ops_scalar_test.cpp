// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "ark/executor.hpp"
#include "ark/model.hpp"
#include "ops_test_common.hpp"
#include "unittest/unittest_utils.h"

#define FACTOR 0.7

template <typename T>
void baseline_scalar_mul(std::vector<void *> &outputs,
                         const std::vector<ark::Dims> &output_shapes,
                         const std::vector<void *> &inputs,
                         const std::vector<ark::Dims> &, int) {
    T *out = static_cast<T *>(outputs[0]);
    T *input = static_cast<T *>(inputs[0]);
    ark::Dims osh = output_shapes[0];
    for (ark::DimType i = 0; i < osh.nelems(); ++i) {
        out[i] = input[i] * T(FACTOR);
    }
};

ark::unittest::State test_scalar_assign_fp16() {
    {
        ark::Model m;
        ark::Tensor t = m.constant(7, ark::Dims(4, 2, 50), ark::FP16);

        ark::DefaultExecutor exe(m);
        exe.compile();

        exe.launch();
        exe.run(1);
        exe.stop();

        std::vector<ark::half_t> data(4 * 2 * 50);
        exe.tensor_read(t, data);
        for (auto v : data) {
            UNITTEST_EQ(v, ark::half_t(7));
        }
    }
    {
        ark::Model m;
        ark::Tensor t = m.tensor(ark::Dims(4, 2, 50), ark::FP16);
        ark::Tensor out = m.copy(7, t);

        ark::DefaultExecutor exe(m);
        exe.compile();

        std::vector<ark::half_t> data(4 * 2 * 50, 3);
        exe.tensor_write(t, data);

        exe.launch();
        exe.run(1);
        exe.stop();

        data.clear();
        data.resize(4 * 2 * 50);
        exe.tensor_read(t, data);
        for (auto v : data) {
            UNITTEST_EQ(v, ark::half_t(7));
        }
    }
    return ark::unittest::SUCCESS;
}

ark::unittest::State test_scalar_mul_fp32() {
    {
        ark::Model m;
        ark::Tensor t = m.tensor(ark::Dims(4, 2, 1), ark::FP32);
        ark::Tensor out = m.mul(t, FACTOR);

        auto result = ark::op_test("scalar_mul_fp32_small", m, {t}, {out},
                                   baseline_scalar_mul<float>);
        UNITTEST_LOG(result);
    }
    {
        ark::Model m;
        ark::Tensor t = m.tensor(ark::Dims(4, 2, 1024), ark::FP32);
        ark::Tensor out = m.mul(t, FACTOR);

        auto result = ark::op_test("scalar_mul_fp32", m, {t}, {out},
                                   baseline_scalar_mul<float>);
        UNITTEST_LOG(result);
        UNITTEST_EQ(result.max_diff[0], 0.0f);
    }
    return ark::unittest::SUCCESS;
}

ark::unittest::State test_scalar_mul_fp16() {
    {
        ark::Model m;
        ark::Tensor t = m.tensor(ark::Dims(4, 2, 1), ark::FP16);
        ark::Tensor out = m.mul(t, FACTOR);

        auto result = ark::op_test("scalar_mul_fp16_small", m, {t}, {out},
                                   baseline_scalar_mul<ark::half_t>);
        UNITTEST_LOG(result);
        UNITTEST_EQ(result.max_diff[0], 0.0f);
    }
    {
        ark::Model m;
        ark::Tensor t = m.tensor(ark::Dims(4, 2, 1024), ark::FP16);
        ark::Tensor out = m.mul(t, FACTOR);

        auto result = ark::op_test("scalar_mul_fp16", m, {t}, {out},
                                   baseline_scalar_mul<ark::half_t>);
        UNITTEST_LOG(result);
        UNITTEST_EQ(result.max_diff[0], 0.0f);
    }
    return ark::unittest::SUCCESS;
}

ark::unittest::State test_scalar_mul_bf16() {
    {
        ark::Model m;
        ark::Tensor t = m.tensor(ark::Dims(4, 2, 1), ark::BF16);
        ark::Tensor out = m.mul(t, FACTOR);

        auto result = ark::op_test("scalar_mul_bf16_small", m, {t}, {out},
                                   baseline_scalar_mul<ark::bfloat16_t>);
        UNITTEST_LOG(result);
        UNITTEST_EQ(result.max_diff[0], 0.0f);
    }
    {
        ark::Model m;
        ark::Tensor t = m.tensor(ark::Dims(4, 2, 1024), ark::BF16);
        ark::Tensor out = m.mul(t, FACTOR);

        auto result = ark::op_test("scalar_mul_bf16", m, {t}, {out},
                                   baseline_scalar_mul<ark::bfloat16_t>);
        UNITTEST_LOG(result);
        UNITTEST_EQ(result.max_diff[0], 0.0f);
    }
    return ark::unittest::SUCCESS;
}

ark::unittest::State test_scalar_mul_invalid() {
    {
        ark::Model m;
        ark::Tensor t = m.tensor(ark::Dims(4, 2, 1024), ark::BF16);
        ark::Tensor out = m.tensor(ark::Dims(4, 2, 1024), ark::FP32);
        UNITTEST_THROW(m.mul(t, 3, out), ark::InvalidUsageError);
    }
    {
        ark::Model m;
        ark::Tensor t = m.tensor(ark::Dims(4, 2, 1024), ark::BF16);
        ark::Tensor out = m.tensor(ark::Dims(4, 4, 1024), ark::BF16);
        UNITTEST_THROW(m.mul(t, 3, out), ark::InvalidUsageError);
    }
    return ark::unittest::SUCCESS;
}

ark::unittest::State test_scalar_mul_fp16_offset() {
    {
        ark::Model m;
        ark::Tensor buf = m.tensor({1024}, ark::FP16);
        ark::Tensor tns = m.refer(buf, {2}, {1024}, {3});
        ark::Tensor out = m.mul(tns, 2, tns);

        ark::DefaultExecutor exe(m);
        exe.compile();

        std::vector<ark::half_t> data(1024, ark::half_t(2));
        exe.tensor_write(buf, data);

        exe.launch();
        exe.run(1);
        exe.stop();

        data.clear();
        data.resize(1024);

        exe.tensor_read(buf, data);

        for (size_t i = 0; i < data.size(); ++i) {
            if (i == 3 || i == 4) {
                UNITTEST_EQ(data[i], 4);
            } else {
                UNITTEST_EQ(data[i], 2);
            }
        }
    }
    return ark::unittest::SUCCESS;
}

ark::unittest::State test_scalar_mul_perf() {
    ark::DimType nelem = 8 * 1024 * 1024;

    ark::Model m;
    ark::Tensor t = m.tensor({nelem}, ark::FP32);
    ark::Tensor out = m.mul(t, 0.7);

    auto result = ark::op_test("scalar_mul_perf", m, {t}, {out},
                               baseline_scalar_mul<float>);
    UNITTEST_LOG(result);
    UNITTEST_EQ(result.max_diff[0], 0.0f);

    float gbps = nelem * sizeof(float) / result.msec_per_iter * 1e-6;
    UNITTEST_LOG(gbps, " GB/s");
    return ark::unittest::SUCCESS;
}

int main() {
    ark::init();
    UNITTEST(test_scalar_assign_fp16);
    UNITTEST(test_scalar_mul_fp32);
    UNITTEST(test_scalar_mul_fp16);
    UNITTEST(test_scalar_mul_bf16);
    UNITTEST(test_scalar_mul_invalid);
    UNITTEST(test_scalar_mul_fp16_offset);
    UNITTEST(test_scalar_mul_perf);
    return ark::unittest::SUCCESS;
}
