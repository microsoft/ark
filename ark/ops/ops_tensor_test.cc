// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "gpu/gpu_buf.h"
#include "gpu/gpu_mgr.h"
#include "include/ark.h"
#include "include/ark_utils.h"
#include "unittest/unittest_utils.h"
#include <cstring>

using namespace std;

ark::unittest::State test_tensor_memcpy()
{
    ark::Model model;
    ark::Dims shape{2, 3, 4, 5};
    ark::Dims ldims{9, 8, 7, 6};
    ark::Dims offs{4, 3, 2, 1};
    ark::Tensor *tns = model.tensor(shape, ark::FP32, nullptr, ldims, offs);

    // Create an executor
    ark::Executor exe{0, 0, 1, model, "test_tensor_memcpy"};
    exe.compile();

    ark::GpuBuf *buf = static_cast<ark::GpuBuf *>(tns->buf->buf);
    UNITTEST_NE(buf, (ark::GpuBuf *)nullptr);
    UNITTEST_EQ(buf->get_bytes(), ldims.size() * sizeof(float));

    // Fill tensor data: {1.0, 2.0, 3.0, ..., 3024.0}
    auto data = ark::utils::range_floats(ldims.size());
    ark::gpu_memcpy(buf, data.get(), ldims.size() * sizeof(float));

    // Copy tensor data from GPU to CPU
    float *res = (float *)malloc(shape.size() * sizeof(float));
    UNITTEST_NE(res, (float *)nullptr);
    memset(res, 0, shape.size() * sizeof(float));
    exe.tensor_memcpy(res, tns, shape.size() * sizeof(float));

    // Validate
    int idx = 0;
    for (int i = 0; i < shape[0]; ++i) {
        for (int j = 0; j < shape[1]; ++j) {
            for (int k = 0; k < shape[2]; ++k) {
                for (int l = 0; l < shape[3]; ++l) {
                    float truth =
                        (i + offs[0]) * (ldims[1] * ldims[2] * ldims[3]) +
                        (j + offs[1]) * (ldims[2] * ldims[3]) +
                        (k + offs[2]) * (ldims[3]) + (l + offs[3]) + 1;
                    UNITTEST_EQ(res[idx], truth);
                    res[idx] *= 2;
                    ++idx;
                }
            }
        }
    }

    // Copy tensor data from CPU to GPU
    exe.tensor_memcpy(tns, res, shape.size() * sizeof(float));

    // Copy all data from GPU to CPU
    float *res2 = (float *)malloc(ldims.size() * sizeof(float));
    UNITTEST_NE(res2, (float *)nullptr);
    ark::gpu_memcpy(res2, buf, ldims.size() * sizeof(float));

    // Validate
    idx = 0;
    for (int i = 0; i < ldims[0]; ++i) {
        for (int j = 0; j < ldims[1]; ++j) {
            for (int k = 0; k < ldims[2]; ++k) {
                for (int l = 0; l < ldims[3]; ++l) {
                    float val = ldims[1] * ldims[2] * ldims[3] * i +
                                ldims[2] * ldims[3] * j + ldims[3] * k + l + 1;
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

ark::unittest::State test_tensor_layout()
{
    ark::Model model;
    // float buf[2][3][4][5];
    ark::Tensor *tns =
        model.tensor({2, 3, 4, 5}, ark::FP32, nullptr, {8, 7, 6, 5});
    // Refer to each value
    ark::Tensor *ref[2][3][4][5];
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 3; ++j) {
            for (int k = 0; k < 4; ++k) {
                for (int l = 0; l < 5; ++l) {
                    ref[i][j][k][l] =
                        model.tensor({1, 1, 1, 1}, ark::FP32, tns->buf,
                                     tns->ldims, {i, j, k, l}, {});
                }
            }
        }
    }

    // Create an executor
    ark::Executor exe{0, 0, 1, model, "test_tensor_layout"};
    exe.compile();

    ark::GpuBuf *buf = static_cast<ark::GpuBuf *>(tns->buf->buf);
    UNITTEST_NE(buf, (ark::GpuBuf *)nullptr);
    UNITTEST_EQ(buf->get_bytes(), 8 * 7 * 6 * 5 * sizeof(float));

    // Fill tensor data: {1.0, 2.0, 3.0, ..., 120.0}
    auto data = ark::utils::range_floats(2 * 3 * 4 * 5);
    exe.tensor_memcpy(tns, data.get(), 2 * 3 * 4 * 5 * sizeof(float));

    // Check reference values
    float ref_val;
    float truth = 1.0f;
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 3; ++j) {
            for (int k = 0; k < 4; ++k) {
                for (int l = 0; l < 5; ++l) {
                    exe.tensor_memcpy(&ref_val, ref[i][j][k][l], sizeof(float));
                    UNITTEST_EQ(ref_val, truth);
                    truth += 1.0f;
                }
            }
        }
    }

    return ark::unittest::SUCCESS;
}

int main()
{
    ark::init();
    UNITTEST(test_tensor_memcpy);
    UNITTEST(test_tensor_layout);
    return ark::unittest::SUCCESS;
}
