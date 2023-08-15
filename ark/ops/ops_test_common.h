// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef ARK_OPS_TEST_COMMON_H_
#define ARK_OPS_TEST_COMMON_H_

#include "include/ark.h"
#include "unittest/unittest_utils.h"
#include <functional>
#include <ostream>
#include <string>

namespace ark {

struct OpsTestResult
{
    std::string test_name;
    int num_warps_per_sm;
    float msec_per_iter;
    std::vector<float> mse;
    std::vector<float> max_err;
};

std::ostream &operator<<(std::ostream &os, const OpsTestResult &result);

/// A function that takes input arrays and returns the ground truth output
/// arrays
using OpsTestBaseline = std::function<void(
    std::vector<void *> &outputs, const std::vector<ark::Dims> &output_tensors,
    const std::vector<void *> &inputs,
    const std::vector<ark::Dims> &input_tensors)>;

OpsTestResult op_test(const std::string &test_name_prefix, Model &model,
                      const std::vector<Tensor *> &inputs,
                      const std::vector<Tensor *> &outputs,
                      OpsTestBaseline baseline, bool print_on_error = false,
                      int num_warps_per_sm = 8);

OpsTestResult op_test_8(const std::string &test_name_prefix, Model &model,
                        const std::vector<Tensor *> &inputs,
                        const std::vector<Tensor *> &outputs,
                        OpsTestBaseline baseline, bool print_on_error = false);

OpsTestResult op_test_16(const std::string &test_name_prefix, Model &model,
                         const std::vector<Tensor *> &inputs,
                         const std::vector<Tensor *> &outputs,
                         OpsTestBaseline baseline, bool print_on_error = false);

OpsTestResult op_test_32(const std::string &test_name_prefix, Model &model,
                         const std::vector<Tensor *> &inputs,
                         const std::vector<Tensor *> &outputs,
                         OpsTestBaseline baseline, bool print_on_error = false);

void op_test_log(const OpsTestResult &result);

} // namespace ark

#endif // ARK_OPS_TEST_COMMON_H_
