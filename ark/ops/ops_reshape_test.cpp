// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include <numeric>

#include "ark/executor.hpp"
#include "ark/model.hpp"
#include "model/model_tensor.hpp"
#include "ops_test_common.hpp"
#include "unittest/unittest_utils.h"

void test_reshape_checker(ark::Model &m, ark::ModelTensorRef t0,
                          ark::ModelTensorRef t1, const std::string &) {
    ark::DefaultExecutor exe(m);
    exe.compile();

    std::vector<float> data_vec(t0->shape().size());
    std::iota(data_vec.begin(), data_vec.end(), 1.0f);
    exe.tensor_write(t0, data_vec);

    std::vector<float> ref_val(t0->shape().size());
    exe.tensor_read(t1, ref_val);
    for (int i = 0; i < t0->shape().size(); ++i) {
        UNITTEST_EQ(ref_val[i], i + 1);
    }
}

ark::unittest::State test_reshape() {
    ark::Model model;
    // float buf[2][3][4][5];
    ark::ModelTensorRef tns0 = model.tensor({2, 3, 4, 5}, ark::FP32);
    ark::ModelTensorRef tns1 = model.reshape(tns0, {5, 4, 3, 2});

    UNITTEST_EQ(tns1->shape(), ark::Dims(5, 4, 3, 2));
    UNITTEST_EQ(tns1->strides(), ark::Dims(5, 4, 3, 2));

    // For preventing optimize-out
    model.noop(tns0);
    model.noop(tns1);

    test_reshape_checker(model, tns0, tns1, "test_reshape");
    return ark::unittest::SUCCESS;
}

ark::unittest::State test_reshape_infer() {
    ark::Model model;
    // float buf[2][3][4][5];
    ark::ModelTensorRef tns0 = model.tensor({2, 3, 4, 5}, ark::FP32);
    ark::ModelTensorRef tns1 = model.reshape(tns0, {-1, 4, 3, 2});

    UNITTEST_EQ(tns1->shape(), ark::Dims(5, 4, 3, 2));
    UNITTEST_EQ(tns1->strides(), ark::Dims(5, 4, 3, 2));

    // For preventing optimize-out
    model.noop(tns0);
    model.noop(tns1);

    test_reshape_checker(model, tns0, tns1, "test_reshape_infer");
    return ark::unittest::SUCCESS;
}

ark::unittest::State test_reshape_allowzero() {
    ark::Model model;
    // float buf[2][3][4][5];
    ark::ModelTensorRef tns0 = model.tensor({2, 3, 4, 5}, ark::FP32);
    ark::ModelTensorRef tns1 = model.reshape(tns0, {5, 3, 0, 2});

    UNITTEST_EQ(tns1->shape(), ark::Dims(5, 3, 4, 2));
    UNITTEST_EQ(tns1->strides(), ark::Dims(5, 3, 4, 2));

    // For preventing optimize-out
    model.noop(tns0);
    model.noop(tns1);

    test_reshape_checker(model, tns0, tns1, "test_reshape_allowzero");
    return ark::unittest::SUCCESS;
}

ark::unittest::State test_reshape_padded() {
    {
        ark::Model model;
        ark::ModelTensorRef tns0 = model.tensor({32, 64}, ark::FP32, {32, 100});
        ark::ModelTensorRef tns1 = model.reshape(tns0, {32, 64});

        UNITTEST_EQ(tns1->shape(), ark::Dims(32, 64));
        UNITTEST_EQ(tns1->strides(), ark::Dims(32, 100));

        // For preventing optimize-out
        model.noop(tns0);
        model.noop(tns1);

        test_reshape_checker(model, tns0, tns1, "test_reshape_padded");
    }
    {
        ark::Model model;
        ark::ModelTensorRef tns0 = model.tensor({32, 64}, ark::FP32, {32, 100});
        ark::ModelTensorRef tns1 = model.reshape(tns0, {32, 32, 2});

        UNITTEST_EQ(tns1->shape(), ark::Dims(32, 32, 2));
        UNITTEST_EQ(tns1->strides(), ark::Dims(32, 50, 2));

        // For preventing optimize-out
        model.noop(tns0);
        model.noop(tns1);

        test_reshape_checker(model, tns0, tns1, "test_reshape_padded");
    }
    {
        ark::Model model;
        ark::ModelTensorRef tns0 =
            model.tensor({3, 8, 4, 64}, ark::FP32, {3, 16, 4, 64});
        ark::ModelTensorRef tns1 = model.reshape(tns0, {3, 32, 64});

        UNITTEST_EQ(tns1->shape(), ark::Dims(3, 32, 64));
        UNITTEST_EQ(tns1->strides(), ark::Dims(3, 64, 64));

        // For preventing optimize-out
        model.noop(tns0);
        model.noop(tns1);

        test_reshape_checker(model, tns0, tns1, "test_reshape_padded");
    }
    {
        ark::Model model;
        ark::ModelTensorRef tns0 =
            model.tensor({2, 3, 4, 5}, ark::FP32, {3, 3, 4, 5});
        ark::ModelTensorRef tns1 = model.reshape(tns0, {120});

        UNITTEST_EQ(tns1->shape(), ark::Dims(120));
        UNITTEST_EQ(tns1->strides(), ark::Dims(180));

        // For preventing optimize-out
        model.noop(tns0);
        model.noop(tns1);

        test_reshape_checker(model, tns0, tns1, "test_reshape_padded");
    }
    {
        ark::Model model;
        ark::ModelTensorRef tns0 =
            model.tensor({2, 3, 4, 5}, ark::FP32, {3, 3, 8, 5});
        ark::ModelTensorRef tns1 = model.reshape(tns0, {6, 20});

        UNITTEST_EQ(tns1->shape(), ark::Dims(6, 20));
        UNITTEST_EQ(tns1->strides(), ark::Dims(9, 40));

        // For preventing optimize-out
        model.noop(tns0);
        model.noop(tns1);

        test_reshape_checker(model, tns0, tns1, "test_reshape_padded");
    }
    {
        ark::Model model;
        ark::ModelTensorRef tns0 =
            model.tensor({4, 3}, ark::FP32, {4, 9}, {0, 5});
        ark::ModelTensorRef tns1 = model.reshape(tns0, {4, 3, 1});

        UNITTEST_EQ(tns1->shape(), ark::Dims(4, 3, 1));
        UNITTEST_EQ(tns1->strides(), ark::Dims(4, 9, 1));
        UNITTEST_EQ(tns1->offsets(), ark::Dims(0, 5, 0));

        // For preventing optimize-out
        model.noop(tns0);
        model.noop(tns1);

        test_reshape_checker(model, tns0, tns1, "test_reshape_padded");
    }
    {
        ark::Model model;
        ark::ModelTensorRef tns0 =
            model.tensor({2, 6, 2}, ark::FP32, {3, 6, 2}, {1, 0, 0});
        ark::ModelTensorRef tns1 = model.reshape(tns0, {8, 3});

        UNITTEST_EQ(tns1->shape(), ark::Dims(8, 3));
        UNITTEST_EQ(tns1->strides(), ark::Dims(12, 3));
        UNITTEST_EQ(tns1->offsets(), ark::Dims(4, 0));

        // For preventing optimize-out
        model.noop(tns0);
        model.noop(tns1);

        test_reshape_checker(model, tns0, tns1, "test_reshape_padded");
    }
    {
        ark::Model model;
        ark::ModelTensorRef tns0 =
            model.tensor({64, 256}, ark::FP32, {512, 512}, {128, 128});
        ark::ModelTensorRef tns1 = model.reshape(tns0, {1, 64, 256});

        UNITTEST_EQ(tns1->shape(), ark::Dims(1, 64, 256));
        UNITTEST_EQ(tns1->strides(), ark::Dims(1, 512, 512));
        UNITTEST_EQ(tns1->offsets(), ark::Dims(0, 128, 128));

        // For preventing optimize-out
        model.noop(tns0);
        model.noop(tns1);

        test_reshape_checker(model, tns0, tns1, "test_reshape_padded");
    }
    {
        ark::Model model;
        ark::ModelTensorRef tns0 =
            model.tensor({1, 64, 256}, ark::FP32, {1, 512, 512}, {0, 128, 128});
        ark::ModelTensorRef tns1 = model.reshape(tns0, {64, 256});

        UNITTEST_EQ(tns1->shape(), ark::Dims(64, 256));
        UNITTEST_EQ(tns1->strides(), ark::Dims(512, 512));
        UNITTEST_EQ(tns1->offsets(), ark::Dims(128, 128));

        // For preventing optimize-out
        model.noop(tns0);
        model.noop(tns1);

        test_reshape_checker(model, tns0, tns1, "test_reshape_padded");
    }
    return ark::unittest::SUCCESS;
}

ark::unittest::State test_reshape_invalid() {
    {
        ark::Model model;
        std::vector<ark::DimType> new_shape = {64, 256};
        UNITTEST_THROW(model.reshape(nullptr, new_shape),
                       ark::InvalidUsageError);
    }
    {
        ark::Model model;
        ark::ModelTensorRef tns = model.tensor({64, 256}, ark::FP32);
        std::vector<ark::DimType> new_shape = {64, -1, -1, 256};
        UNITTEST_THROW(model.reshape(tns, new_shape), ark::InvalidUsageError);
    }
    {
        ark::Model model;
        ark::ModelTensorRef tns = model.tensor({64, 256}, ark::FP32);
        std::vector<ark::DimType> new_shape = {128, -3};
        UNITTEST_THROW(model.reshape(tns, new_shape), ark::InvalidUsageError);
    }
    {
        ark::Model model;
        ark::ModelTensorRef tns = model.tensor({64, 256}, ark::FP32);
        std::vector<ark::DimType> new_shape = {32, -1, 0};
        UNITTEST_THROW(model.reshape(tns, new_shape), ark::InvalidUsageError);
    }
    {
        ark::Model model;
        ark::ModelTensorRef tns = model.tensor({64, 256}, ark::FP32);
        std::vector<ark::DimType> new_shape = {32, -1, 0};
        UNITTEST_THROW(model.reshape(tns, new_shape), ark::InvalidUsageError);
    }
    {
        ark::Model model;
        ark::ModelTensorRef tns = model.tensor({64, 256}, ark::FP32);
        std::vector<ark::DimType> new_shape = {3, -1};
        UNITTEST_THROW(model.reshape(tns, new_shape), ark::InvalidUsageError);
    }
    {
        ark::Model model;
        ark::ModelTensorRef tns = model.tensor({64, 256}, ark::FP32);
        std::vector<ark::DimType> new_shape = {1024};
        UNITTEST_THROW(model.reshape(tns, new_shape), ark::InvalidUsageError);
    }
    {
        ark::Model model;
        UNITTEST_THROW(model.reshape(nullptr, {64, 256}),
                       ark::InvalidUsageError);
    }
    {
        ark::Model model;
        ark::ModelTensorRef tns = model.tensor({64, 256}, ark::FP32);
        UNITTEST_THROW(model.reshape(tns, {}), ark::InvalidUsageError);
    }
    {
        ark::Model model;
        ark::ModelTensorRef tns = model.tensor({64, 256}, ark::FP32, {64, 512});
        UNITTEST_THROW(model.reshape(tns, {16384}), ark::ModelError);
    }
    return ark::unittest::SUCCESS;
}

int main() {
    ark::init();
    UNITTEST(test_reshape);
    UNITTEST(test_reshape_infer);
    UNITTEST(test_reshape_allowzero);
    UNITTEST(test_reshape_padded);
    UNITTEST(test_reshape_invalid);
    return ark::unittest::SUCCESS;
}
