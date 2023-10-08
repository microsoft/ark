// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "include/ark.h"
#include "include/ark_utils.h"
#include "ops_test_common.h"
#include "unittest/unittest_utils.h"

#define SCALE_FACTOR 0.7

template <typename T>
void baseline_scale(std::vector<void *> &outputs,
                    const std::vector<ark::Dims> &output_shapes,
                    const std::vector<void *> &inputs,
                    const std::vector<ark::Dims> &, int)
{
    T *out = static_cast<T *>(outputs[0]);
    T *input = static_cast<T *>(inputs[0]);
    ark::Dims osh = output_shapes[0];
    for (ark::DimType i = 0; i < osh.size(); ++i) {
        out[i] = input[i] * T(SCALE_FACTOR);
    }
};

ark::unittest::State test_scale_fp16()
{
    ark::Model m;
    ark::Tensor *t = m.tensor(ark::Dims(4, 2, 1024), ark::FP16);
    ark::Tensor *out = m.scale(t, SCALE_FACTOR);

    auto result =
        ark::op_test("scale_fp16", m, {t}, {out}, baseline_scale<ark::half_t>);
    ark::op_test_log(result);
    return ark::unittest::SUCCESS;
}

int main()
{
    ark::init();
    UNITTEST(test_scale_fp16);
    return ark::unittest::SUCCESS;
}
