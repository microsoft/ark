// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "include/ark.h"
#include "include/ark_utils.h"
#include "unittest/unittest_utils.h"

ark::unittest::State test_identity()
{
    ark::Model model;
    // float buf[2][3][4][5];
    ark::Tensor *tns0 = model.tensor({2, 3, 4, 5}, ark::FP32);
    ark::Tensor *tns1 = model.identity(tns0);

    // Create an executor
    ark::Executor exe{0, 1, model, "test_tensor_layout"};
    exe.compile();

    int num_elem = 2 * 3 * 4 * 5;

    // Fill tensor data: {1.0, 2.0, 3.0, ..., 120.0}
    auto data = ark::utils::range_floats(num_elem);
    tns0->write(data.get());

    // Check identity values
    std::vector<float> ref_val(num_elem);
    tns1->read(ref_val.data());
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
