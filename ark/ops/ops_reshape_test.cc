// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "include/ark.h"
#include "include/ark_utils.h"
#include "unittest/unittest_utils.h"

using namespace std;

ark::unittest::State test_reshape() {
    ark::Model model;
    // float buf[2][3][4][5];
    ark::Tensor *tns0 = model.tensor({2, 3, 4, 5}, ark::FP32);
    ark::Tensor *tns1 = model.reshape(tns0, {5, 4, 3, 2});

    // Create an executor
    ark::Executor exe{0, 1, model, "test_tensor_layout"};
    exe.compile();

    int num_elem = 2 * 3 * 4 * 5;

    UNITTEST_EQ(tns1->shape, ark::Dims(5, 4, 3, 2));

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

ark::unittest::State test_reshape_infer() {
    ark::Model model;
    // float buf[2][3][4][5];
    ark::Tensor *tns0 = model.tensor({2, 3, 4, 5}, ark::FP32);
    ark::Tensor *tns1 = model.reshape(tns0, {-1, 4, 3, 2});

    // Create an executor
    ark::Executor exe{0, 1, model, "test_tensor_layout"};
    exe.compile();

    int num_elem = 2 * 3 * 4 * 5;

    UNITTEST_EQ(tns1->shape, ark::Dims(5, 4, 3, 2));

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

ark::unittest::State test_reshape_allowzero() {
    ark::Model model;
    // float buf[2][3][4][5];
    ark::Tensor *tns0 = model.tensor({2, 3, 4, 5}, ark::FP32);
    ark::Tensor *tns1 = model.reshape(tns0, {5, 3, 0, 2});

    // Create an executor
    ark::Executor exe{0, 1, model, "test_tensor_layout"};
    exe.compile();

    int num_elem = 2 * 3 * 4 * 5;

    UNITTEST_EQ(tns1->shape, ark::Dims(5, 3, 4, 2));

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

int main() {
    ark::init();
    UNITTEST(test_reshape);
    UNITTEST(test_reshape_infer);
    UNITTEST(test_reshape_allowzero);
    return ark::unittest::SUCCESS;
}
