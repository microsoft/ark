// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef ARK_UTILS_H
#define ARK_UTILS_H

#include <array>
#include <iostream>
#include <list>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>
// clang-format off
#include "vector_types.h"
#include "cutlass/half.h"
// clang-format on
// typedef uint16_t half_t;
typedef cutlass::half_t half_t;

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

#endif
