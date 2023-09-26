// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "include/ark.h"
#include "include/ark_utils.h"
#include "ops_test_common.h"
#include "unittest/unittest_utils.h"
#include <cmath>

using namespace std;

template <typename T>
void baseline_rmsnorm(std::vector<void *> &outputs,
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

ark::unittest::State test_rmsnorm_fp32()
{
    ark::Model m;
    ark::Tensor *t = m.tensor(ark::Dims(1, 8192), ark::FP32);
    ark::Tensor *out = m.rmsnorm(t);
    auto result =
        ark::op_test("rmsnorm_fp32", m, {t}, {out}, baseline_rmsnorm<float>);
    ark::op_test_log(result);
    return ark::unittest::SUCCESS;
}

ark::unittest::State test_rmsnorm_fp16()
{
    ark::Model model;
    ark::Tensor *input = model.tensor(ark::Dims(1, 8192), ark::FP16);
    ark::Tensor *output = model.rmsnorm(input);
    
    // std::vector<ark::half_t> data;
    // for (int i = 0; i < 8192; ++i) {
    //     data.push_back(ark::half_t(8.0f));
    // }
    auto result = ark::op_test("rmsnorm_fp16", model, {input}, {output},
                               baseline_rmsnorm<ark::half_t>);
    ark::op_test_log(result);
    return ark::unittest::SUCCESS;
}

ark::unittest::State test_rmsnorm_compare()
{
    ark::srand();

    ark::Dims shape(2048, 16384);
    auto input_data_fp16 = ark::utils::rand_halfs(shape.size(), 0.1);
    auto input_data_fp32 = std::unique_ptr<float[]>(new float[shape.size()]);
    for (int i = 0; i < shape.size(); ++i) {
        input_data_fp32[i] = float(input_data_fp16[i]);
    }

    std::vector<float> output_fp32(shape.size());
    std::vector<ark::half_t> output_fp16_tmp(shape.size());
    std::vector<float> output_fp16(shape.size());

    std::string test_name = "rmsnorm_compare";
    int num_warps_per_sm = 16;

    {
        ark::Model model_fp32;
        ark::Tensor *input = model_fp32.tensor(shape, ark::FP32);
        ark::Tensor *output = model_fp32.rmsnorm(input);

        ark::Executor exe{0, 1, model_fp32, test_name, num_warps_per_sm};
        exe.compile();

        input->write(input_data_fp32.get());

        exe.launch();
        exe.run(1);
        exe.stop();

        output->read(output_fp32.data());
    }

    {
        ark::Model model_fp16;
        ark::Tensor *input = model_fp16.tensor(shape, ark::FP16);
        ark::Tensor *output = model_fp16.rmsnorm(input);

        ark::Executor exe{0, 1, model_fp16, test_name, num_warps_per_sm};
        exe.compile();

        input->write(input_data_fp16.get());

        exe.launch();
        exe.run(1);
        exe.stop();

        output->read(output_fp16_tmp.data());
    }

    for (ark::DimType i = 0; i < shape.size(); ++i) {
        output_fp16[i] = float(output_fp16_tmp[i]);
    }

    auto comp = ark::tensor_compare(output_fp32.data(), output_fp16.data(), shape);

    ark::OpsTestResult result;
    result.test_name = test_name;
    result.num_warps_per_sm = num_warps_per_sm;
    result.mse.push_back(comp.mse);
    result.max_diff.push_back(comp.max_diff);
    result.max_err_rate.push_back(comp.max_error_rate);

    ark::op_test_log(result);
    return ark::unittest::SUCCESS;
}

int main()
{
    ark::init();
    // UNITTEST(test_rmsnorm_fp32);
    // UNITTEST(test_rmsnorm_fp16);
    UNITTEST(test_rmsnorm_compare);
    return ark::unittest::SUCCESS;
}
