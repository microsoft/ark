// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "include/ark.h"
#include "include/ark_utils.h"
#include "ops_test_common.h"
#include "unittest/unittest_utils.h"

template <typename FromType, typename ToType>
void baseline_cast(std::vector<void *> &outputs,
                   const std::vector<ark::Dims> &output_shapes,
                   const std::vector<void *> &inputs,
                   const std::vector<ark::Dims> &) {
    ToType *out = static_cast<ToType *>(outputs[0]);
    FromType *input = static_cast<FromType *>(inputs[0]);
    ark::Dims osh = output_shapes[0];
    for (ark::DimType i = 0; i < osh.size(); ++i) {
        out[i] = ToType(input[i]);
    }
};

template <typename ToType>
void baseline_cast_from_byte(std::vector<void *> &outputs,
                             const std::vector<ark::Dims> &output_shapes,
                             const std::vector<void *> &inputs,
                             const std::vector<ark::Dims> &) {
    ToType *out = static_cast<ToType *>(outputs[0]);
    // input is a byte array, but force read it as ToType.
    ToType *input = reinterpret_cast<ToType *>(inputs[0]);
    ark::Dims osh = output_shapes[0];
    for (ark::DimType i = 0; i < osh.size(); ++i) {
        out[i] = input[i];
    }
};

template <typename FromType>
void baseline_cast_to_byte(std::vector<void *> &outputs,
                           const std::vector<ark::Dims> &,
                           const std::vector<void *> &inputs,
                           const std::vector<ark::Dims> &input_shapes) {
    // output is a byte array, but force write it as FromType.
    FromType *out = reinterpret_cast<FromType *>(outputs[0]);
    FromType *input = static_cast<FromType *>(inputs[0]);
    ark::Dims ish = input_shapes[0];
    for (ark::DimType i = 0; i < ish.size(); ++i) {
        out[i] = input[i];
    }
};

ark::unittest::State test_cast_fp16_to_fp32() {
    ark::Model m;
    ark::Tensor *t = m.tensor(ark::Dims(4, 2, 1024), ark::FP16);
    ark::Tensor *out = m.cast(t, ark::FP32);

    auto result = ark::op_test("cast_fp16_to_fp32", m, {t}, {out},
                               baseline_cast<ark::half_t, float>);
    UNITTEST_LOG(result);
    UNITTEST_EQ(result.max_diff[0], 0.0f);
    return ark::unittest::SUCCESS;
}

ark::unittest::State test_cast_fp16_to_int32() {
    ark::Model m;
    ark::Tensor *t = m.tensor(ark::Dims(4, 2, 1024), ark::FP16);
    ark::Tensor *out = m.cast(t, ark::INT32);

    std::vector<ark::half_t> input_data(t->shape.size());
    for (size_t i = 0; i < input_data.size(); ++i) {
        input_data[i] = ark::half_t((i + 1) % 1000);
    }

    auto result =
        ark::op_test("cast_fp16_to_int32", m, {t}, {out},
                     baseline_cast<ark::half_t, int>, {input_data.data()});
    UNITTEST_LOG(result);
    UNITTEST_EQ(result.max_diff[0], 0.0f);
    return ark::unittest::SUCCESS;
}

ark::unittest::State test_cast_fp32_to_fp16() {
    ark::Model m;
    ark::Tensor *t = m.tensor(ark::Dims(4, 2, 1024), ark::FP32);
    ark::Tensor *out = m.cast(t, ark::FP16);

    auto result = ark::op_test("cast_fp32_to_fp16", m, {t}, {out},
                               baseline_cast<float, ark::half_t>);
    UNITTEST_LOG(result);
    UNITTEST_EQ(result.max_diff[0], 0.0f);
    return ark::unittest::SUCCESS;
}

ark::unittest::State test_cast_fp32_to_int32() {
    ark::Model m;
    ark::Tensor *t = m.tensor(ark::Dims(4, 2, 1024), ark::FP32);
    ark::Tensor *out = m.cast(t, ark::INT32);

    std::vector<float> input_data(t->shape.size());
    for (size_t i = 0; i < input_data.size(); ++i) {
        input_data[i] = float((i + 1) % 1000);
    }

    auto result = ark::op_test("cast_fp32_to_int32", m, {t}, {out},
                               baseline_cast<float, int>, {input_data.data()});
    UNITTEST_LOG(result);
    UNITTEST_EQ(result.max_diff[0], 0.0f);
    return ark::unittest::SUCCESS;
}

ark::unittest::State test_cast_int32_to_fp32() {
    ark::Model m;
    ark::Tensor *t = m.tensor(ark::Dims(4, 2, 1024), ark::INT32);
    ark::Tensor *out = m.cast(t, ark::FP32);

    std::vector<int> input_data(t->shape.size());
    for (size_t i = 0; i < input_data.size(); ++i) {
        input_data[i] = (i + 1) % 1000;
    }

    auto result = ark::op_test("cast_int32_to_fp32", m, {t}, {out},
                               baseline_cast<int, float>, {input_data.data()});
    UNITTEST_LOG(result);
    UNITTEST_EQ(result.max_diff[0], 0.0f);
    return ark::unittest::SUCCESS;
}

ark::unittest::State test_cast_int32_to_fp16() {
    ark::Model m;
    ark::Tensor *t = m.tensor(ark::Dims(4, 2, 1024), ark::INT32);
    ark::Tensor *out = m.cast(t, ark::FP16);

    std::vector<int> input_data(t->shape.size());
    for (size_t i = 0; i < input_data.size(); ++i) {
        input_data[i] = (i + 1) % 1000;
    }

    auto result =
        ark::op_test("cast_int32_to_fp16", m, {t}, {out},
                     baseline_cast<int, ark::half_t>, {input_data.data()});
    UNITTEST_LOG(result);
    UNITTEST_EQ(result.max_diff[0], 0.0f);
    return ark::unittest::SUCCESS;
}

ark::unittest::State test_cast_byte_to_fp32() {
    ark::Model m;
    ark::Tensor *t = m.tensor(ark::Dims(4, 2, 1024), ark::BYTE);
    ark::Tensor *out = m.cast(t, ark::FP32);

    std::vector<char> input_data(t->shape.size());
    for (size_t i = 0; i < input_data.size(); ++i) {
        input_data[i] = (i + 1) % 256;
    }

    auto result =
        ark::op_test("cast_byte_to_fp32", m, {t}, {out},
                     baseline_cast_from_byte<float>, {input_data.data()});
    UNITTEST_LOG(result);
    UNITTEST_EQ(result.max_diff[0], 0.0f);
    return ark::unittest::SUCCESS;
}

ark::unittest::State test_cast_byte_to_fp16() {
    ark::Model m;
    ark::Tensor *t = m.tensor(ark::Dims(4, 2, 1024), ark::BYTE);
    ark::Tensor *out = m.cast(t, ark::FP16);

    std::vector<char> input_data(t->shape.size());
    for (size_t i = 0; i < input_data.size(); ++i) {
        input_data[i] = (i + 1) % 256;
    }

    auto result =
        ark::op_test("cast_byte_to_fp16", m, {t}, {out},
                     baseline_cast_from_byte<ark::half_t>, {input_data.data()});
    UNITTEST_LOG(result);
    UNITTEST_EQ(result.max_diff[0], 0.0f);
    return ark::unittest::SUCCESS;
}

ark::unittest::State test_cast_byte_to_int32() {
    ark::Model m;
    ark::Tensor *t = m.tensor(ark::Dims(4, 2, 1024), ark::BYTE);
    ark::Tensor *out = m.cast(t, ark::INT32);

    std::vector<char> input_data(t->shape.size());
    for (size_t i = 0; i < input_data.size(); ++i) {
        input_data[i] = (i + 1) % 256;
    }

    auto result =
        ark::op_test("cast_byte_to_int32", m, {t}, {out},
                     baseline_cast_from_byte<float>, {input_data.data()});
    UNITTEST_LOG(result);
    UNITTEST_EQ(result.max_diff[0], 0.0f);
    return ark::unittest::SUCCESS;
}

ark::unittest::State test_cast_fp32_to_byte() {
    ark::Model m;
    ark::Tensor *t = m.tensor(ark::Dims(4, 2, 1024), ark::FP32);
    ark::Tensor *out = m.cast(t, ark::BYTE);

    auto result = ark::op_test("cast_fp32_to_byte", m, {t}, {out},
                               baseline_cast_to_byte<float>);
    UNITTEST_LOG(result);
    UNITTEST_EQ(result.max_diff[0], 0.0f);
    return ark::unittest::SUCCESS;
}

ark::unittest::State test_cast_fp16_to_byte() {
    ark::Model m;
    ark::Tensor *t = m.tensor(ark::Dims(4, 2, 1024), ark::FP16);
    ark::Tensor *out = m.cast(t, ark::BYTE);

    auto result = ark::op_test("cast_fp16_to_byte", m, {t}, {out},
                               baseline_cast_to_byte<ark::half_t>);
    UNITTEST_LOG(result);
    UNITTEST_EQ(result.max_diff[0], 0.0f);
    return ark::unittest::SUCCESS;
}

ark::unittest::State test_cast_int32_to_byte() {
    ark::Model m;
    ark::Tensor *t = m.tensor(ark::Dims(4, 2, 1024), ark::INT32);
    ark::Tensor *out = m.cast(t, ark::BYTE);

    auto result = ark::op_test("cast_int32_to_byte", m, {t}, {out},
                               baseline_cast_to_byte<int>);
    UNITTEST_LOG(result);
    UNITTEST_EQ(result.max_diff[0], 0.0f);
    return ark::unittest::SUCCESS;
}

ark::unittest::State test_cast_bf16_to_float() {
    ark::Model m;
    ark::Tensor *t = m.tensor(ark::Dims(4, 2, 1024), ark::BF16);
    ark::Tensor *out = m.cast(t, ark::FP32);

    auto result = ark::op_test("cast_bf16_to_float", m, {t}, {out},
                               baseline_cast<ark::bfloat16_t, float>);
    UNITTEST_LOG(result);
    UNITTEST_EQ(result.max_diff[0], 0.0f);
    return ark::unittest::SUCCESS;
}

ark::unittest::State test_cast_float_to_bf16() {
    ark::Model m;
    ark::Tensor *t = m.tensor(ark::Dims(4, 2, 1024), ark::FP32);
    ark::Tensor *out = m.cast(t, ark::BF16);

    auto result = ark::op_test("cast_float_to_bf16", m, {t}, {out},
                               baseline_cast<float, ark::bfloat16_t>);
    UNITTEST_LOG(result);
    UNITTEST_EQ(result.max_diff[0], 0.0f);
    return ark::unittest::SUCCESS;
}

int main() {
    ark::init();
    UNITTEST(test_cast_fp16_to_fp32);
    UNITTEST(test_cast_fp16_to_int32);
    UNITTEST(test_cast_fp32_to_fp16);
    UNITTEST(test_cast_fp32_to_int32);
    UNITTEST(test_cast_int32_to_fp32);
    UNITTEST(test_cast_int32_to_fp16);
    UNITTEST(test_cast_byte_to_fp32);
    UNITTEST(test_cast_byte_to_fp16);
    UNITTEST(test_cast_byte_to_int32);
    UNITTEST(test_cast_fp32_to_byte);
    UNITTEST(test_cast_fp16_to_byte);
    UNITTEST(test_cast_int32_to_byte);
    UNITTEST(test_cast_bf16_to_float);
    UNITTEST(test_cast_float_to_bf16);
    return ark::unittest::SUCCESS;
}
