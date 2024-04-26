// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include <cstring>
#include <numeric>

#include "ark/executor.hpp"
#include "ark/model.hpp"
#include "model/model_tensor.hpp"
#include "unittest/unittest_utils.h"

using namespace std;

ark::unittest::State test_tensor_strides() {
    ark::Model model;
    ark::Dims shape{4, 1};
    ark::Dims strides{4, 2};
    auto tns = model.tensor(shape, ark::FP32, strides);

    // For preventing optimize-out
    model.noop(tns);

    // Create an executor
    ark::DefaultExecutor exe(model);
    exe.compile();

    // Fill buffer data: {1.0, 2.0, 3.0, 4.0}
    std::vector<float> data(shape.size());
    std::iota(data.begin(), data.end(), 1.0f);
    exe.tensor_write(tns, data);

    // Copy tensor data from GPU to CPU
    std::vector<float> res(shape.size(), 0.0f);
    exe.tensor_read(tns, res);

    // Validate
    for (auto i = 0; i < shape.size(); ++i) {
        UNITTEST_EQ(res[i], i + 1);
    }

    return ark::unittest::SUCCESS;
}

ark::unittest::State test_tensor_memcpy() {
    ark::Model model;
    ark::Dims shape{2, 3, 4, 5};
    ark::Dims strides{9, 8, 7, 6};
    ark::Dims offs{4, 3, 2, 1};
    auto buffer = model.tensor(strides, ark::FP32);
    auto tns = model.refer(buffer, shape, strides, offs);

    // For preventing optimize-out
    model.noop(buffer);
    model.noop(tns);

    // Create an executor
    ark::DefaultExecutor exe(model);
    exe.compile();

    // Fill buffer data: {1.0, 2.0, 3.0, ..., 3024.0}
    std::vector<float> data(strides.size());
    std::iota(data.begin(), data.end(), 1.0f);
    exe.tensor_write(buffer, data);

    // Copy tensor data from GPU to CPU
    std::vector<float> res(shape.size(), 0.0f);
    exe.tensor_read(tns, res);

    // Validate
    int idx = 0;
    for (int i = 0; i < shape[0]; ++i) {
        for (int j = 0; j < shape[1]; ++j) {
            for (int k = 0; k < shape[2]; ++k) {
                for (int l = 0; l < shape[3]; ++l) {
                    float truth =
                        (i + offs[0]) * (strides[1] * strides[2] * strides[3]) +
                        (j + offs[1]) * (strides[2] * strides[3]) +
                        (k + offs[2]) * (strides[3]) + (l + offs[3]) + 1;
                    UNITTEST_EQ(res[idx], truth);
                    // Modify the value
                    res[idx] *= 2;
                    ++idx;
                }
            }
        }
    }

    // Copy modified tensor data from CPU to GPU
    exe.tensor_write(tns, res);

    // Copy buffer data from GPU to CPU
    std::vector<float> res2(strides.size(), -1);
    exe.tensor_read(buffer, res2);

    // Validate
    idx = 0;
    for (int i = 0; i < strides[0]; ++i) {
        for (int j = 0; j < strides[1]; ++j) {
            for (int k = 0; k < strides[2]; ++k) {
                for (int l = 0; l < strides[3]; ++l) {
                    float val = strides[1] * strides[2] * strides[3] * i +
                                strides[2] * strides[3] * j + strides[3] * k +
                                l + 1;
                    if (i >= offs[0] && i < offs[0] + shape[0] &&
                        j >= offs[1] && j < offs[1] + shape[1] &&
                        k >= offs[2] && k < offs[2] + shape[2] &&
                        l >= offs[3] && l < offs[3] + shape[3]) {
                        val *= 2;
                    }
                    UNITTEST_EQ(res2[idx], val);
                    idx++;
                }
            }
        }
    }

    return ark::unittest::SUCCESS;
}

ark::unittest::State test_tensor_layout() {
    ark::Model model;
    // float buf[2][3][4][5];
    ark::ModelTensorRef tns =
        model.tensor({2, 3, 4, 5}, ark::FP32, {8, 7, 6, 5});
    // For preventing optimize-out
    model.noop(tns);
    // Refer to each value
    ark::ModelTensorRef ref[2][3][4][5];
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 3; ++j) {
            for (int k = 0; k < 4; ++k) {
                for (int l = 0; l < 5; ++l) {
                    ref[i][j][k][l] = model.refer(tns, {1, 1, 1, 1},
                                                  tns->strides(), {i, j, k, l});
                    // For preventing optimize-out
                    model.noop(ref[i][j][k][l]);
                }
            }
        }
    }

    // Create an executor
    ark::DefaultExecutor exe(model);
    exe.compile();

    // Fill tensor data: {1.0, 2.0, 3.0, ..., 120.0}
    std::vector<float> data(2 * 3 * 4 * 5);
    std::iota(data.begin(), data.end(), 1.0f);
    exe.tensor_write(tns, data);

    // Check reference values
    std::vector<float> ref_val(1);
    float truth = 1.0f;
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 3; ++j) {
            for (int k = 0; k < 4; ++k) {
                for (int l = 0; l < 5; ++l) {
                    exe.tensor_read(ref[i][j][k][l], ref_val);
                    UNITTEST_EQ(ref_val[0], truth);
                    truth += 1.0f;
                }
            }
        }
    }

    return ark::unittest::SUCCESS;
}

int main() {
    ark::init();
    UNITTEST(test_tensor_strides);
    UNITTEST(test_tensor_memcpy);
    UNITTEST(test_tensor_layout);
    return ark::unittest::SUCCESS;
}
