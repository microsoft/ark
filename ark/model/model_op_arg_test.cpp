// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "model_op_arg.hpp"

#include "ark/model.hpp"
#include "model_tensor.hpp"
#include "unittest/unittest_utils.h"

ark::unittest::State test_oparg() {
    {
        int data = 7;
        auto arg = ark::ModelOpArg(data);
        UNITTEST_EQ(arg.type_name(), "INT");

        int get_data = arg.value<int>();
        UNITTEST_EQ(get_data, data);

        auto arg2 = ark::ModelOpArg(arg);
        UNITTEST_EQ(arg2.type_name(), "INT");

        get_data = arg2.value<int>();
        UNITTEST_EQ(get_data, data);
    }

    {
        int64_t data = 7;
        auto arg = ark::ModelOpArg(data);
        UNITTEST_EQ(arg.type_name(), "INT64");

        int64_t get_data = arg.value<int64_t>();
        UNITTEST_EQ(get_data, data);

        auto arg2 = ark::ModelOpArg(arg);
        UNITTEST_EQ(arg2.type_name(), "INT64");

        get_data = arg2.value<int64_t>();
        UNITTEST_EQ(get_data, data);
    }

    {
        uint64_t data = 7;
        auto arg = ark::ModelOpArg(data);
        UNITTEST_EQ(arg.type_name(), "UINT64");

        uint64_t get_data = arg.value<uint64_t>();
        UNITTEST_EQ(get_data, data);

        auto arg2 = ark::ModelOpArg(arg);
        UNITTEST_EQ(arg2.type_name(), "UINT64");

        get_data = arg2.value<uint64_t>();
        UNITTEST_EQ(get_data, data);
    }

    {
        bool data = true;
        auto arg = ark::ModelOpArg(data);
        UNITTEST_EQ(arg.type_name(), "BOOL");

        bool get_data = arg.value<bool>();
        UNITTEST_EQ(get_data, data);

        auto arg2 = ark::ModelOpArg(arg);
        UNITTEST_EQ(arg2.type_name(), "BOOL");

        get_data = arg2.value<bool>();
        UNITTEST_EQ(get_data, data);
    }

    {
        float data = 7;
        auto arg = ark::ModelOpArg(data);
        UNITTEST_EQ(arg.type_name(), "FLOAT");

        float get_data = arg.value<float>();
        UNITTEST_EQ(get_data, data);

        auto arg2 = ark::ModelOpArg(arg);
        UNITTEST_EQ(arg2.type_name(), "FLOAT");

        get_data = arg2.value<float>();
        UNITTEST_EQ(get_data, data);
    }

    {
        ark::Dims data(1, 2, 3, 4);
        auto arg = ark::ModelOpArg(data);
        UNITTEST_EQ(arg.type_name(), "DIMS");

        ark::Dims get_data = arg.value<ark::Dims>();
        UNITTEST_EQ(get_data, data);

        auto arg2 = ark::ModelOpArg(arg);
        UNITTEST_EQ(arg2.type_name(), "DIMS");

        get_data = arg2.value<ark::Dims>();
        UNITTEST_EQ(get_data, data);
    }

    {
        ark::Model m;
        auto data = m.tensor({1, 2, 3, 4}, ark::FP32);
        auto arg = ark::ModelOpArg(data);
        UNITTEST_EQ(arg.type_name(), "TENSOR");

        ark::ModelTensorRef get_data = arg.value<ark::ModelTensorRef>();
        UNITTEST_NE(get_data, nullptr);
        UNITTEST_EQ(get_data->shape(), data->shape());

        auto arg2 = ark::ModelOpArg(arg);
        UNITTEST_EQ(arg2.type_name(), "TENSOR");

        get_data = arg2.value<ark::ModelTensorRef>();
        UNITTEST_NE(get_data, nullptr);
        UNITTEST_EQ(get_data->shape(), data->shape());
    }

    return ark::unittest::SUCCESS;
}

int main() {
    ark::init();
    UNITTEST(test_oparg);
    return 0;
}
