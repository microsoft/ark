// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "ark/gpu/gpu_kernel.h"
#include "ark/include/ark.h"
#include "ark/include/ark_utils.h"
#include "ark/logging.h"
#include "ark/unittest/unittest_utils.h"

using namespace std;

//
void test_reduce_internal(unsigned int n, unsigned int m, unsigned int k,
                          ark::DimType axis, bool is_relu = false)
{
    size_t buf_x_sz = (size_t)m * (size_t)n * (size_t)k * sizeof(ark::half_t);
    size_t buf_y_sz = (size_t)m * (size_t)n * sizeof(ark::half_t);

    // Set data.
    ark::srand();
    auto data_a = ark::utils::rand_halfs(buf_x_sz / sizeof(ark::half_t), 0.01);

    // Copy the ground truth results into CPU memory.
    void *gt = malloc(buf_y_sz);
    UNITTEST_NE(gt, (void *)nullptr);

    for (unsigned int i = 0; i < n; ++i) {
        for (unsigned int j = 0; j < m; ++j) {
            ark::half_t v = 0;
            for (unsigned int l = 0; l < k; ++l) {
                int idx;
                if (axis == 0) {
                    idx = i * m + j + l * m * n;
                } else if (axis == 1) {
                    idx = i * m * k + j + l * m;
                } else if (axis == 2) {
                    idx = i * m * k + j * k + l;
                } else {
                    assert(false);
                }
                ark::half_t x = data_a[idx];
                v += x;
            }
            if (is_relu) {
                if ((float)v < 0) {
                    v = 0;
                }
            }
            ((ark::half_t *)gt)[i * m + j] = v;
        }
    }

    //
    ark::Model model;
    ark::Tensor *tns_x = nullptr;
    ark::Tensor *tns_y = nullptr;
    if (axis == 0) {
        tns_x = model.tensor({k, n, m}, ark::FP16);
        tns_y = model.tensor({1, n, m}, ark::FP16);
    } else if (axis == 1) {
        tns_x = model.tensor({n, k, m}, ark::FP16);
        tns_y = model.tensor({n, 1, m}, ark::FP16);
    } else if (axis = 2) {
        tns_x = model.tensor({n, m, k}, ark::FP16);
        tns_y = model.tensor({n, m, 1}, ark::FP16);
    } else {
        LOGERR("invalid axis");
    }

    model.reduce(tns_x, axis, tns_y, is_relu);

    //
    ark::Executor exe{0, 0, 1, model, "test_reduce"};
    exe.compile();

    // Set data.
    exe.tensor_memcpy(tns_x, data_a.get(), buf_x_sz);

    exe.launch();
    exe.run(1);
    exe.stop();

    // Copy results of the loop kernel routine into CPU memory.
    void *res = malloc(buf_y_sz);
    UNITTEST_NE(res, (void *)nullptr);
    exe.tensor_memcpy(res, tns_y, buf_y_sz);

    // Compare results with the ground truth.
    auto p =
        ark::utils::cmp_matrix((ark::half_t *)gt, (ark::half_t *)res, m, n);
    float max_err = p.second;
    LOG(ark::INFO, "reduce:", n, 'x', m, 'x', k, "(relu=", is_relu, ") ",
        setprecision(4), " mse ", p.first, " max_err ", max_err * 100, "%");

    free(res);
    free(gt);

    UNITTEST_EQ(max_err, 0.0);
}

ark::unittest::State test_reduce()
{
    // TODO: implement reduce for axis = 0 and axis = 1
    for (int axis = 2; axis < 3; axis++) {
        test_reduce_internal(1, 64, 2, axis);
        test_reduce_internal(1, 64, 8, axis);
        test_reduce_internal(1, 64, 9, axis);
        test_reduce_internal(2, 64, 4, axis);
        test_reduce_internal(8, 64, 4, axis);
        test_reduce_internal(64, 64, 4, axis);
        test_reduce_internal(1, 256, 256, axis);
        test_reduce_internal(1024, 384, 4, axis);
    }

    return ark::unittest::SUCCESS;
}

int main()
{
    ark::init();
    UNITTEST(test_reduce);
    return ark::unittest::SUCCESS;
}
