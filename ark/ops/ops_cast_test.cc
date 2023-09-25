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
                   const std::vector<ark::Dims> &)
{
    ToType *out = static_cast<ToType *>(outputs[0]);
    FromType *input = static_cast<FromType *>(inputs[0]);
    ark::Dims osh = output_shapes[0];
    for (ark::DimType i = 0; i < osh.size(); ++i) {
        out[i] = ToType(input[i]);
    }
};

ark::unittest::State test_cast_fp16_to_fp32()
{
    ark::Model m;
    ark::Tensor *t = m.tensor(ark::Dims(4, 2, 1024), ark::FP16);
    ark::Tensor *out = m.cast(t, ark::FP32);

    auto result = ark::op_test("cast_fp16_to_fp32", m, {t}, {out},
                               baseline_cast<ark::half_t, float>);
    ark::op_test_log(result);
    return ark::unittest::SUCCESS;
}

ark::unittest::State test_cast_fp32_to_fp16()
{
    ark::Model m;
    ark::Tensor *t = m.tensor(ark::Dims(4, 2, 1024), ark::FP32);
    ark::Tensor *out = m.cast(t, ark::FP16);

    auto result = ark::op_test("cast_fp32_to_fp16", m, {t}, {out},
                               baseline_cast<float, ark::half_t>);
    ark::op_test_log(result);
    return ark::unittest::SUCCESS;
}

int main()
{
    ark::init();
    UNITTEST(test_cast_fp16_to_fp32);
    UNITTEST(test_cast_fp32_to_fp16);
    return ark::unittest::SUCCESS;
}
