// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include <cmath>

#include "include/ark.h"
#include "include/ark_utils.h"
#include "ops_test_common.h"
#include "unittest/unittest_utils.h"

using namespace std;

template <typename T>
void baseline_rmsnorm(std::vector<void *> &outputs,
                      const std::vector<ark::Dims> &output_shapes,
                      const std::vector<void *> &inputs,
                      const std::vector<ark::Dims> &input_shapes) {
    T *out = static_cast<T *>(outputs[0]);
    T *input = static_cast<T *>(inputs[0]);

    ark::Dims osh = output_shapes[0].dims4();
    ark::Dims ish = input_shapes[0].dims4();

    for (ark::DimType n = 0; n < ish[0]; ++n) {
        for (ark::DimType c = 0; c < ish[1]; ++c) {
            for (ark::DimType h = 0; h < ish[2]; ++h) {
                float square_sum = 0;
                for (ark::DimType w = 0; w < ish[3]; ++w) {
                    float val =
                        float(input[n * ish[1] * ish[2] * ish[3] +
                                    c * ish[2] * ish[3] + h * ish[3] + w]);
                    square_sum += val * val;
                }
                float eps = 1e-5;
                float rms = std::sqrt(square_sum / ish[3]) + eps;
                for (ark::DimType w = 0; w < ish[3]; ++w) {
                    out[n * osh[1] * osh[2] * osh[3] + c * osh[2] * osh[3] +
                        h * osh[3] + w] =
                        T(float(input[n * osh[1] * osh[2] * osh[3] +
                                      c * osh[2] * osh[3] + h * osh[3] + w]) /
                          rms);
                }
            }
        }
    }
}

ark::unittest::State test_rmsnorm_fp32() {
    ark::Model m;
    ark::Tensor *t = m.tensor(ark::Dims(1, 32, 32, 8192), ark::FP32);
    ark::Tensor *out = m.rmsnorm(t);
    auto result =
        ark::op_test("rmsnorm_fp32", m, {t}, {out}, baseline_rmsnorm<float>);
    UNITTEST_LOG(result);
    return ark::unittest::SUCCESS;
}

ark::unittest::State test_rmsnorm_fp16() {
    ark::Model model;
    ark::Tensor *input = model.tensor(ark::Dims(1, 32, 32, 8192), ark::FP16);
    ark::Tensor *output = model.rmsnorm(input);
    auto result = ark::op_test("rmsnorm_fp16", model, {input}, {output},
                               baseline_rmsnorm<ark::half_t>);
    UNITTEST_LOG(result);
    return ark::unittest::SUCCESS;
}

int main() {
    ark::init();
    UNITTEST(test_rmsnorm_fp32);
    UNITTEST(test_rmsnorm_fp16);
    return ark::unittest::SUCCESS;
}
