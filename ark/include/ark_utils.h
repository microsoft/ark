// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef ARK_UTILS_H
#define ARK_UTILS_H

#include "ark.h"
#include <functional>
#include <memory>
#include <vector>

namespace ark {

// 16-bit floating point type.
struct alignas(2) half_t
{
    uint16_t storage;
    half_t() = default;
    // Constructor with float parameter
    half_t(float f);
    // Conversion operator from half to float
    operator float() const;
};

} // namespace ark

ark::half_t operator+(ark::half_t const &lhs, ark::half_t const &rhs);
ark::half_t operator-(ark::half_t const &lhs, ark::half_t const &rhs);
ark::half_t operator*(ark::half_t const &lhs, ark::half_t const &rhs);
ark::half_t &operator+=(ark::half_t &lhs, ark::half_t const &rhs);
ark::half_t &operator-=(ark::half_t &lhs, ark::half_t const &rhs);

ark::half_t abs(ark::half_t const &val);

// A set of utility functions
namespace ark {
namespace utils {

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

// Return an array where each element is 0.
template <typename T> std::unique_ptr<T[]> zeros(size_t num)
{
    T *ret = new T[num];
    for (size_t i = 0; i < num; ++i) {
        ret[i] = T(0);
    }
    return std::unique_ptr<T[]>(ret);
}

// Return an array where each element is 1.
template <typename T> std::unique_ptr<T[]> ones(size_t num)
{
    T *ret = new T[num];
    for (size_t i = 0; i < num; ++i) {
        ret[i] = T(1);
    }
    return std::unique_ptr<T[]>(ret);
}

// Return the error rate between two values.
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

// Print a matrix.
void print_matrix(half_t *val, unsigned int m, unsigned int n, unsigned int bs,
                  unsigned int lm, unsigned int ln);
void print_matrix(float *val, unsigned int m, unsigned int n, unsigned int bs,
                  unsigned int lm, unsigned int ln);

//
std::pair<float, float> tensor_compare(half_t *ground_truth, half_t *res,
                                       Dims shape, bool print);
std::pair<float, float> tensor_compare(float *ground_truth, float *res,
                                       Dims shape, bool print);

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
