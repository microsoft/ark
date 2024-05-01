// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef ARK_OPS_TEST_COMMON_HPP_
#define ARK_OPS_TEST_COMMON_HPP_

#include <functional>
#include <ostream>
#include <string>

#include "ark/model.hpp"
#include "ark/model_ref.hpp"
#include "bfloat16.h"
#include "half.h"
#include "unittest/unittest_utils.h"

namespace ark {

struct TensorCompareResult {
    float mse;
    float max_diff;
    float max_error_rate;
    size_t num_wrong;
    size_t num_total;
};

/// Calculate the error rate between two values.
/// @tparam T Type of the values
/// @param a First value
/// @param b Second value
/// @return The error rate
template <typename T>
float error_rate(T a, T b) {
    T diff = std::abs(a - b);
    T max = std::max(std::abs(a), std::abs(b));
    if (max == static_cast<T>(0)) {
        return 0;
    }
    return (float)diff / (float)max;
}

/// Return mean squared error and max error rate between two tensors.
/// @tparam T data type of the tensors.
/// @param ground_truth ground truth data array.
/// @param res input data array to compare with the ground truth.
/// @param shape shape of the tensor.
/// @param print whether to print wrong values.
/// @return a pair of mean squared error and max error rate.
template <typename T>
TensorCompareResult tensor_compare(T *ground_truth, T *res, Dims shape,
                                   bool print = false) {
    DimType nelem = shape.nelems();
    int ndims = shape.ndims();
    float l2_loss = 0;
    float max_err = 0;
    float max_diff = 0;
    size_t num_wrong = 0;
    for (DimType i = 0; i < nelem; ++i) {
        float diff = (float)(ground_truth[i] - res[i]);
        if (std::abs(diff) > max_diff) {
            max_diff = std::abs(diff);
        }
        l2_loss += diff * diff;

        float err = error_rate(ground_truth[i], res[i]);
        if (err > 0.) {
            num_wrong++;
            if (print) {
                Dims idx(std::vector<DimType>(ndims, 0));
                for (int j = 0; j < ndims; ++j) {
                    DimType vol = 1;
                    for (int k = j + 1; k < ndims; ++k) {
                        vol *= shape[k];
                    }
                    idx[j] = (i / vol) % shape[j];
                }
                std::cout << idx << " expected " << ground_truth[i]
                          << ", actually " << res[i] << " (err: " << err << ")"
                          << std::endl;
            }
            if (err > max_err) {
                max_err = err;
            }
        }
    }
    TensorCompareResult result;
    result.mse = l2_loss / nelem;
    result.max_diff = max_diff;
    result.max_error_rate = max_err;
    result.num_wrong = num_wrong;
    result.num_total = nelem;
    return result;
}

struct OpsTestResult {
    std::string test_name;
    int num_warps_per_sm;
    int iter;
    float msec_per_iter;
    std::vector<float> mse;
    std::vector<float> max_diff;
    std::vector<float> max_err_rate;
    std::vector<size_t> num_wrong;
    std::vector<size_t> num_total;
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

#if defined(ARK_CUDA)
#define DEFAULT_OP_TEST_NUM_WARPS_PER_SM 16
#else  // !defined(ARK_CUDA)
#define DEFAULT_OP_TEST_NUM_WARPS_PER_SM 8
#endif  // !defined(ARK_CUDA)

class Model;

OpsTestResult op_test(const std::string &test_name_prefix, const Model &model,
                      const std::vector<Tensor> &inputs,
                      const std::vector<Tensor> &outputs,
                      OpsTestBaseline baseline,
                      const std::vector<void *> &inputs_data = {},
                      bool print_on_error = false, int rank = 0,
                      int world_size = 1,
                      int num_warps_per_sm = DEFAULT_OP_TEST_NUM_WARPS_PER_SM);

OpsTestResult op_test_8(const std::string &test_name_prefix, const Model &model,
                        const std::vector<Tensor> &inputs,
                        const std::vector<Tensor> &outputs,
                        OpsTestBaseline baseline,
                        const std::vector<void *> &inputs_data = {},
                        bool print_on_error = false, int rank = 0,
                        int world_size = 1);

OpsTestResult op_test_16(const std::string &test_name_prefix,
                         const Model &model, const std::vector<Tensor> &inputs,
                         const std::vector<Tensor> &outputs,
                         OpsTestBaseline baseline,
                         const std::vector<void *> &inputs_data = {},
                         bool print_on_error = false, int rank = 0,
                         int world_size = 1);

OpsTestResult op_test_32(const std::string &test_name_prefix,
                         const Model &model, const std::vector<Tensor> &inputs,
                         const std::vector<Tensor> &outputs,
                         OpsTestBaseline baseline,
                         const std::vector<void *> &inputs_data = {},
                         bool print_on_error = false, int rank = 0,
                         int world_size = 1);

OpsTestGpuMem to_gpu(void *host_ptr, size_t size);

void *from_gpu(const OpsTestGpuMem &test_gpu_mem, void *host_ptr = nullptr);

void sync_gpu();

}  // namespace ark

#endif  // ARK_OPS_TEST_COMMON_HPP_
