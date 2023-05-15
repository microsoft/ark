// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "ark/include/ark.h"
#include "ark/ops/ops_test_utils.h"
#include "ark/unittest/unittest_utils.h"

using namespace std;

ark::unittest::State test_identity()
{
    ark::Model model;
    // float buf[2][3][4][5];
    ark::Tensor *tns0 = model.tensor({2, 3, 4, 5}, ark::FP32);
    ark::Tensor *tns1 = model.identity(tns0);

    // Create an executor
    ark::Executor exe{0, 0, 1, model, "test_tensor_layout"};
    exe.compile();

    int num_elem = 2 * 3 * 4 * 5;

    ark::GpuBuf *buf0 = exe.get_gpu_buf(tns0);
    ark::GpuBuf *buf1 = exe.get_gpu_buf(tns1);
    UNITTEST_NE(buf0, (ark::GpuBuf *)nullptr);
    UNITTEST_NE(buf1, (ark::GpuBuf *)nullptr);
    UNITTEST_EQ(buf0->get_bytes(), num_elem * sizeof(float));
    UNITTEST_EQ(buf1->get_bytes(), num_elem * sizeof(float));

    // Fill tensor data: {1.0, 2.0, 3.0, ..., 120.0}
    auto data = range_floats(num_elem);
    exe.tensor_memcpy(tns0, data.get(), num_elem * sizeof(float));

    // Check identity values
    float *ref_val = new float[num_elem];
    exe.tensor_memcpy(ref_val, tns1, num_elem * sizeof(float));
    for (int i = 0; i < num_elem; ++i) {
        UNITTEST_EQ(ref_val[i], (float)(i + 1));
    }

    return ark::unittest::SUCCESS;
}

int main()
{
    ark::init();
    UNITTEST(test_identity);
    return ark::unittest::SUCCESS;
}
