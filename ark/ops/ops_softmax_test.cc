// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "include/ark.h"
#include "include/ark_utils.h"
#include "ops_test_common.h"
#include "unittest/unittest_utils.h"
#include <cmath>

template <typename T>
void baseline_softmax(std::vector<void *> &outputs,
                               const std::vector<ark::Dims> &output_shapes,
                               const std::vector<void *> &inputs,
                               const std::vector<ark::Dims> &input_shapes)
{
    T *out = static_cast<T *>(outputs[0]);
    T *input = static_cast<T *>(inputs[0]);

    ark::Dims osh = output_shapes[0].dims4();
    ark::Dims ish = input_shapes[0].dims4();

    for (ark::DimType n = 0; n < ish[0]; ++n) {
        for (ark::DimType c = 0; c < ish[1]; ++c) {
            for (ark::DimType h = 0; h < ish[2]; ++h) {
                T sum = 0;
                for (ark::DimType w = 0; w < ish[3]; ++w) {
                    sum += std::exp(input[w + h * ish[3] + c * ish[2] * ish[3] +
                                          n * ish[1] * ish[2] * ish[3]]);
                }
                for (ark::DimType w = 0; w < ish[3]; ++w) {
                    out[w + h * osh[3] + c * osh[2] * osh[3] +
                        n * osh[1] * osh[2] * osh[3]] =
                        std::exp(input[w + h * ish[3] + c * ish[2] * ish[3] +
                                       n * ish[1] * ish[2] * ish[3]]) /
                        sum;
                }
            }
        }
    }
};

ark::unittest::State test_softmax_fp32()
{
    ark::Model m;
    ark::Tensor *t = m.tensor(ark::Dims(2, 8192), ark::FP32);
    ark::Tensor *out = m.softmax(t);

    auto result = ark::op_test("reduce_axis3", m, {t}, {out},
                               baseline_softmax<float>);
    ark::op_test_log(result);
    return ark::unittest::SUCCESS;
}


int main()
{
    ark::init();
    UNITTEST(test_softmax_fp32);
    return ark::unittest::SUCCESS;
}
