// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef ARK_OPS_TEST_COMMON_H_
#define ARK_OPS_TEST_COMMON_H_

#include <functional>
#include <ostream>
#include <string>

#include "include/ark.h"
#include "include/ark_utils.h"
#include "unittest/unittest_utils.h"

namespace ark {

struct TensorCompareResult {
    float mse;
    float max_diff;
    float max_error_rate;
};

TensorCompareResult tensor_compare(half_t *ground_truth, half_t *res,
                                   Dims shape, bool print = false);

/// Return mean squared error and max error rate between two float tensors.
/// @param ground_truth ground truth data array.
/// @param res input data array to compare with the ground truth.
/// @param shape shape of the tensor.
/// @param print whether to print wrong values.
/// @return a pair of mean squared error and max error rate.
TensorCompareResult tensor_compare(float *ground_truth, float *res, Dims shape,
                                   bool print = false);

/// Return mean squared error and max error rate between two int tensors.
/// @param ground_truth ground truth data array.
/// @param res input data array to compare with the ground truth.
/// @param shape shape of the tensor.
/// @param print whether to print wrong values.
/// @return a pair of mean squared error and max error rate.
TensorCompareResult tensor_compare(int *ground_truth, int *res, Dims shape,
                                   bool print = false);

/// Return mean squared error and max error rate between two byte tensors.
/// @param ground_truth ground truth data array.
/// @param res input data array to compare with the ground truth.
/// @param shape shape of the tensor.
/// @param print whether to print wrong values.
/// @return a pair of mean squared error and max error rate.
TensorCompareResult tensor_compare(uint8_t *ground_truth, uint8_t *res,
                                   Dims shape, bool print = false);

struct OpsTestResult {
    std::string test_name;
    int num_warps_per_sm;
    float msec_per_iter;
    std::vector<float> mse;
    std::vector<float> max_diff;
    std::vector<float> max_err_rate;
};

std::ostream &operator<<(std::ostream &os, const OpsTestResult &result);

class OpsTestGpuMem {
   public:
    OpsTestGpuMem(size_t size);
    ~OpsTestGpuMem();
    void *get() const;
    size_t size() const;

   private:
    size_t size_;
    void *gpu_ptr_;
};

/// A function that takes input arrays and returns the ground truth output
/// arrays
using OpsTestBaseline = std::function<void(
    std::vector<void *> &outputs, const std::vector<ark::Dims> &output_tensors,
    const std::vector<void *> &inputs,
    const std::vector<ark::Dims> &input_tensors, int rank)>;

OpsTestResult op_test(const std::string &test_name_prefix, Model &model,
                      const std::vector<Tensor *> &inputs,
                      const std::vector<Tensor *> &outputs,
                      OpsTestBaseline baseline,
                      const std::vector<void *> &inputs_data = {},
                      bool print_on_error = false, int rank = 0,
                      int world_size = 1, int num_warps_per_sm = 8);

OpsTestResult op_test_8(const std::string &test_name_prefix, Model &model,
                        const std::vector<Tensor *> &inputs,
                        const std::vector<Tensor *> &outputs,
                        OpsTestBaseline baseline,
                        const std::vector<void *> &inputs_data = {},
                        bool print_on_error = false, int rank = 0,
                        int world_size = 1);

OpsTestResult op_test_16(const std::string &test_name_prefix, Model &model,
                         const std::vector<Tensor *> &inputs,
                         const std::vector<Tensor *> &outputs,
                         OpsTestBaseline baseline,
                         const std::vector<void *> &inputs_data = {},
                         bool print_on_error = false, int rank = 0,
                         int world_size = 1);

OpsTestResult op_test_32(const std::string &test_name_prefix, Model &model,
                         const std::vector<Tensor *> &inputs,
                         const std::vector<Tensor *> &outputs,
                         OpsTestBaseline baseline,
                         const std::vector<void *> &inputs_data = {},
                         bool print_on_error = false, int rank = 0,
                         int world_size = 1);

OpsTestGpuMem to_gpu(const void *host_ptr, size_t size);

void *from_gpu(const OpsTestGpuMem &test_gpu_mem, void *host_ptr = nullptr);

void sync_gpu();

}  // namespace ark

#endif  // ARK_OPS_TEST_COMMON_H_
