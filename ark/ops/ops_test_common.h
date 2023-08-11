// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef ARK_OPS_TEST_COMMON_H_
#define ARK_OPS_TEST_COMMON_H_

#include "include/ark.h"
#include "unittest/unittest_utils.h"
#include <functional>
#include <string>

// TODO: deprecate this
void test_bcast_fp32(std::string op_name, ark::DimType bs, ark::DimType n,
                     ark::DimType m, bool overwrite = false);
// TODO: deprecate this
void test_bcast_fp16(std::string op_name, ark::DimType bs, ark::DimType n,
                     ark::DimType m, bool overwrite = false);

namespace ark {

struct OpsTestResult
{
    std::vector<float> mse;
    std::vector<float> max_err;
    float elapsed_msec;
};

/// A function that takes input arrays and returns the ground truth output
/// arrays
using OpsTestBaseline = std::function<void(std::vector<void *> &outputs,
                                           const std::vector<void *> &inputs)>;

OpsTestResult op_test(const std::string &test_name, Model &model,
                      const std::vector<Tensor *> &inputs,
                      const std::vector<Tensor *> &outputs,
                      OpsTestBaseline baseline, int num_warps_per_sm,
                      bool print_on_error = false);

OpsTestResult op_test_8(const std::string &test_name, Model &model,
                        const std::vector<Tensor *> &inputs,
                        const std::vector<Tensor *> &outputs,
                        OpsTestBaseline baseline, bool print_on_error = false);

OpsTestResult op_test_16(const std::string &test_name, Model &model,
                         const std::vector<Tensor *> &inputs,
                         const std::vector<Tensor *> &outputs,
                         OpsTestBaseline baseline, bool print_on_error = false);

OpsTestResult op_test_32(const std::string &test_name, Model &model,
                         const std::vector<Tensor *> &inputs,
                         const std::vector<Tensor *> &outputs,
                         OpsTestBaseline baseline, bool print_on_error = false);

} // namespace ark

#endif // ARK_OPS_TEST_COMMON_H_
