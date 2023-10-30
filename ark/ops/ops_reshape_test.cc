// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include <numeric>

#include "include/ark.h"
#include "unittest/unittest_utils.h"

void test_reshape_checker(ark::Model &m, ark::Tensor *t0, ark::Tensor *t1,
                          const std::string &test_name) {
    ark::Executor exe{0, 1, m, test_name};
    exe.compile();

    std::vector<float> data_vec(t0->shape.size());
    std::iota(data_vec.begin(), data_vec.end(), 1.0f);
    t0->write(data_vec.data());

    std::vector<float> ref_val(t0->shape.size());
    t1->read(ref_val.data());
    for (int i = 0; i < t0->shape.size(); ++i) {
        UNITTEST_EQ(ref_val[i], i + 1);
    }
}

ark::unittest::State test_reshape() {
    ark::Model model;
    // float buf[2][3][4][5];
    ark::Tensor *tns0 = model.tensor({2, 3, 4, 5}, ark::FP32);
    ark::Tensor *tns1 = model.reshape(tns0, {5, 4, 3, 2});

    UNITTEST_EQ(tns1->shape, ark::Dims(5, 4, 3, 2));
    UNITTEST_EQ(tns1->ldims, ark::Dims(5, 4, 3, 2));

    test_reshape_checker(model, tns0, tns1, "test_reshape");
    return ark::unittest::SUCCESS;
}

ark::unittest::State test_reshape_infer() {
    ark::Model model;
    // float buf[2][3][4][5];
    ark::Tensor *tns0 = model.tensor({2, 3, 4, 5}, ark::FP32);
    ark::Tensor *tns1 = model.reshape(tns0, {-1, 4, 3, 2});

    UNITTEST_EQ(tns1->shape, ark::Dims(5, 4, 3, 2));
    UNITTEST_EQ(tns1->ldims, ark::Dims(5, 4, 3, 2));

    test_reshape_checker(model, tns0, tns1, "test_reshape_infer");
    return ark::unittest::SUCCESS;
}

ark::unittest::State test_reshape_allowzero() {
    ark::Model model;
    // float buf[2][3][4][5];
    ark::Tensor *tns0 = model.tensor({2, 3, 4, 5}, ark::FP32);
    ark::Tensor *tns1 = model.reshape(tns0, {5, 3, 0, 2});

    UNITTEST_EQ(tns1->shape, ark::Dims(5, 3, 4, 2));
    UNITTEST_EQ(tns1->ldims, ark::Dims(5, 3, 4, 2));

    test_reshape_checker(model, tns0, tns1, "test_reshape_allowzero");
    return ark::unittest::SUCCESS;
}

ark::unittest::State test_reshape_padded() {
    {
        ark::Model model;
        ark::Tensor *tns0 =
            model.tensor({32, 64}, ark::FP32, nullptr, {32, 100});
        ark::Tensor *tns1 = model.reshape(tns0, {32, 64});

        UNITTEST_EQ(tns1->shape, ark::Dims(32, 64));
        UNITTEST_EQ(tns1->ldims, ark::Dims(32, 100));
        test_reshape_checker(model, tns0, tns1, "test_reshape_padded");
    }
    {
        ark::Model model;
        ark::Tensor *tns0 =
            model.tensor({32, 64}, ark::FP32, nullptr, {32, 100});
        ark::Tensor *tns1 = model.reshape(tns0, {32, 32, 2});

        UNITTEST_EQ(tns1->shape, ark::Dims(32, 32, 2));
        UNITTEST_EQ(tns1->ldims, ark::Dims(32, 50, 2));
        test_reshape_checker(model, tns0, tns1, "test_reshape_padded");
    }
    {
        ark::Model model;
        ark::Tensor *tns0 =
            model.tensor({3, 8, 4, 64}, ark::FP32, nullptr, {3, 16, 4, 64});
        ark::Tensor *tns1 = model.reshape(tns0, {3, 32, 64});

        UNITTEST_EQ(tns1->shape, ark::Dims(3, 32, 64));
        UNITTEST_EQ(tns1->ldims, ark::Dims(3, 64, 64));
        test_reshape_checker(model, tns0, tns1, "test_reshape_padded");
    }
    {
        ark::Model model;
        ark::Tensor *tns0 =
            model.tensor({2, 3, 4, 5}, ark::FP32, nullptr, {3, 3, 4, 5});
        ark::Tensor *tns1 = model.reshape(tns0, {120});

        UNITTEST_EQ(tns1->shape, ark::Dims(120));
        UNITTEST_EQ(tns1->ldims, ark::Dims(180));
        test_reshape_checker(model, tns0, tns1, "test_reshape_padded");
    }
    {
        ark::Model model;
        ark::Tensor *tns0 =
            model.tensor({2, 3, 4, 5}, ark::FP32, nullptr, {3, 3, 8, 5});
        ark::Tensor *tns1 = model.reshape(tns0, {6, 20});

        UNITTEST_EQ(tns1->shape, ark::Dims(6, 20));
        UNITTEST_EQ(tns1->ldims, ark::Dims(9, 40));
        test_reshape_checker(model, tns0, tns1, "test_reshape_padded");
    }
    {
        ark::Model model;
        ark::Tensor *tns0 =
            model.tensor({4, 3}, ark::FP32, nullptr, {4, 9}, {0, 5});
        ark::Tensor *tns1 = model.reshape(tns0, {4, 3, 1});

        UNITTEST_EQ(tns1->shape, ark::Dims(4, 3, 1));
        UNITTEST_EQ(tns1->ldims, ark::Dims(4, 9, 1));
        UNITTEST_EQ(tns1->offs, ark::Dims(0, 5, 0));
        test_reshape_checker(model, tns0, tns1, "test_reshape_padded");
    }
    {
        ark::Model model;
        ark::Tensor *tns0 =
            model.tensor({2, 6, 2}, ark::FP32, nullptr, {3, 6, 2}, {1, 0, 0});
        ark::Tensor *tns1 = model.reshape(tns0, {8, 3});

        UNITTEST_EQ(tns1->shape, ark::Dims(8, 3));
        UNITTEST_EQ(tns1->ldims, ark::Dims(12, 3));
        UNITTEST_EQ(tns1->offs, ark::Dims(4, 0));
        test_reshape_checker(model, tns0, tns1, "test_reshape_padded");
    }
    {
        ark::Model model;
        ark::Tensor *tns0 =
            model.tensor({64, 256}, ark::FP32, nullptr, {512, 512}, {128, 128});
        ark::Tensor *tns1 = model.reshape(tns0, {1, 64, 256});

        UNITTEST_EQ(tns1->shape, ark::Dims(1, 64, 256));
        UNITTEST_EQ(tns1->ldims, ark::Dims(1, 512, 512));
        UNITTEST_EQ(tns1->offs, ark::Dims(0, 128, 128));
        test_reshape_checker(model, tns0, tns1, "test_reshape_padded");
    }
    {
        ark::Model model;
        ark::Tensor *tns0 = model.tensor({1, 64, 256}, ark::FP32, nullptr,
                                         {1, 512, 512}, {0, 128, 128});
        ark::Tensor *tns1 = model.reshape(tns0, {64, 256});

        UNITTEST_EQ(tns1->shape, ark::Dims(64, 256));
        UNITTEST_EQ(tns1->ldims, ark::Dims(512, 512));
        UNITTEST_EQ(tns1->offs, ark::Dims(128, 128));
        test_reshape_checker(model, tns0, tns1, "test_reshape_padded");
    }
    return ark::unittest::SUCCESS;
}

int main() {
    ark::init();
    UNITTEST(test_reshape);
    UNITTEST(test_reshape_infer);
    UNITTEST(test_reshape_allowzero);
    UNITTEST(test_reshape_padded);
    return ark::unittest::SUCCESS;
}
