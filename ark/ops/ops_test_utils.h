// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef ARK_OPS_TEST_UTILS_H_
#define ARK_OPS_TEST_UTILS_H_

#include <iostream>
#include <memory>
#include <utility>

// clang-format off
#include "vector_types.h"
#include "cutlass/half.h"
// clang-format on
#include "ark/gpu/gpu_buf.h"
#include "ark/include/ark.h"

// Return a random value array.
template <typename T> std::unique_ptr<T[]> rand_array(size_t num, float max_val)
{
    int mid = RAND_MAX / 2;
    T *ret = new T[num];
    for (size_t i = 0; i < num; ++i) {
        ret[i] = T((ark::rand() - mid) / (float)mid * max_val);
    }
    return std::unique_ptr<T[]>(ret);
}
//
float error_rate(half_t a, half_t b);
float error_rate(float a, float b);

// Return mean squared error and max error rate between two matrices.
std::pair<float, float> cmp_matrix(half_t *ground_truth, half_t *res,
                                   unsigned int m, unsigned int n,
                                   unsigned int bs = 1, unsigned int lm = 0,
                                   unsigned int ln = 0, bool print = false);
std::pair<float, float> cmp_matrix(float *ground_truth, float *res,
                                   unsigned int m, unsigned int n,
                                   unsigned int bs = 1, unsigned int lm = 0,
                                   unsigned int ln = 0, bool print = false);

//
void print_matrix(half_t *val, unsigned int m, unsigned int n, unsigned int bs,
                  unsigned int lm, unsigned int ln);
void print_matrix(float *val, unsigned int m, unsigned int n, unsigned int bs,
                  unsigned int lm, unsigned int ln);

//
std::string get_kernel_code(const std::string &name);

// Return mean squared error and max error rate between two matrices.
template <typename T>
std::pair<float, float> tensor_compare(T *ground_truth, T *res, ark::Dims shape,
                                       bool print = false)
{
    ark::DimType nelem = shape.size();
    int ndims = shape.ndims();
    float l2_loss = 0;
    float max_err = 0;
    for (ark::DimType i = 0; i < nelem; ++i) {
        float diff = (float)(ground_truth[i] - res[i]);
        l2_loss += diff * diff;

        float err = error_rate(ground_truth[i], res[i]);
        if (err > 0.) {
            if (print) {
                ark::Dims idx;
                for (int j = 0; j < ndims; ++j) {
                    ark::DimType vol = 1;
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
    return {l2_loss / nelem, max_err};
}

#endif // ARK_OPS_TEST_UTILS_H_
