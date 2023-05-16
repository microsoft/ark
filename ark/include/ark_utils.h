// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef ARK_UTILS_H
#define ARK_UTILS_H
#include "ark.h"
#include <array>
#include <cassert>
#include <cstring>
#include <functional>
#include <iostream>
#include <list>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace ark {

typedef uint16_t half_t;
namespace utils {
// Return an array of range values.
template <typename T>
std::unique_ptr<T[]> range_array(size_t num, float begin, float diff);
// Return a random half_t array.
std::unique_ptr<half_t[]> rand_halfs(size_t num, float max_val);
// Return a random float array.
std::unique_ptr<float[]> rand_floats(size_t num, float max_val);
// Return a half_t range array.
std::unique_ptr<half_t[]> range_halfs(size_t num, float begin = 1.0f,
                                      float diff = 1.0f);
// Return a float range array.
std::unique_ptr<float[]> range_floats(size_t num, float begin = 1.0f,
                                      float diff = 1.0f);

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

float half2float(half_t h);

half_t float2half(float f);

// Spawn a process that runs `func`. Returns PID of the spawned process.
int proc_spawn(const std::function<int()> &func);
// Wait for a spawned process with PID `pid`.
// Return -1 on any unexpected failures, otherwise return the exit status.
int proc_wait(int pid);
// Wait for multiple child processes.
// Return 0 on success, -1 on any unexpected failure, otherwise the first seen
// non-zero exit status.
int proc_wait(const std::vector<int> &pids);
} // namespace utils
} // namespace ark

#endif
