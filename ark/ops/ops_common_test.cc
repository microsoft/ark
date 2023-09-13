// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "include/ark.h"
#include "ops_common.h"
#include "unittest/unittest_utils.h"

ark::unittest::State test_oparg()
{
    {
        int data = 7;
        auto arg = ark::OpArg(data);
        UNITTEST_EQ(arg.type, ark::OP_ARG_INT);

        int get_data = 0;
        arg.get(&get_data);
        UNITTEST_EQ(get_data, data);

        auto arg2 = ark::OpArg(arg);
        UNITTEST_EQ(arg2.type, ark::OP_ARG_INT);

        get_data = 0;
        arg2.get(&get_data);
        UNITTEST_EQ(get_data, data);
    }

    {
        long long int data = 7;
        auto arg = ark::OpArg(data);
        UNITTEST_EQ(arg.type, ark::OP_ARG_INT64);

        long long int get_data = 0;
        arg.get(&get_data);
        UNITTEST_EQ(get_data, data);

        auto arg2 = ark::OpArg(arg);
        UNITTEST_EQ(arg2.type, ark::OP_ARG_INT64);

        get_data = 0;
        arg2.get(&get_data);
        UNITTEST_EQ(get_data, data);
    }

    {
        uint64_t data = 7;
        auto arg = ark::OpArg(data);
        UNITTEST_EQ(arg.type, ark::OP_ARG_UINT64);

        uint64_t get_data = 0;
        arg.get(&get_data);
        UNITTEST_EQ(get_data, data);

        auto arg2 = ark::OpArg(arg);
        UNITTEST_EQ(arg2.type, ark::OP_ARG_UINT64);

        get_data = 0;
        arg2.get(&get_data);
        UNITTEST_EQ(get_data, data);
    }

    {
        bool data = true;
        auto arg = ark::OpArg(data);
        UNITTEST_EQ(arg.type, ark::OP_ARG_BOOL);

        bool get_data = false;
        arg.get(&get_data);
        UNITTEST_EQ(get_data, data);

        auto arg2 = ark::OpArg(arg);
        UNITTEST_EQ(arg2.type, ark::OP_ARG_BOOL);

        get_data = false;
        arg2.get(&get_data);
        UNITTEST_EQ(get_data, data);
    }

    {
        float data = 7;
        auto arg = ark::OpArg(data);
        UNITTEST_EQ(arg.type, ark::OP_ARG_FLOAT);

        float get_data = 0;
        arg.get(&get_data);
        UNITTEST_EQ(get_data, data);

        auto arg2 = ark::OpArg(arg);
        UNITTEST_EQ(arg2.type, ark::OP_ARG_FLOAT);

        get_data = 0;
        arg2.get(&get_data);
        UNITTEST_EQ(get_data, data);
    }

    {
        ark::Dims data(1, 2, 3, 4);
        auto arg = ark::OpArg(data);
        UNITTEST_EQ(arg.type, ark::OP_ARG_DIMS);

        ark::Dims get_data;
        arg.get(&get_data);
        UNITTEST_EQ(get_data, data);

        auto arg2 = ark::OpArg(arg);
        UNITTEST_EQ(arg2.type, ark::OP_ARG_DIMS);

        get_data = ark::Dims();
        arg2.get(&get_data);
        UNITTEST_EQ(get_data, data);
    }

    {
        ark::Model m;
        auto data = m.tensor({1, 2, 3, 4}, ark::FP32);
        auto arg = ark::OpArg(data);
        UNITTEST_EQ(arg.type, ark::OP_ARG_TENSOR);

        ark::Tensor *get_data = nullptr;
        arg.get(&get_data);
        UNITTEST_NE(get_data, (ark::Tensor *)nullptr);
        UNITTEST_EQ(get_data->shape, data->shape);

        auto arg2 = ark::OpArg(arg);
        UNITTEST_EQ(arg2.type, ark::OP_ARG_TENSOR);

        get_data = nullptr;
        arg2.get(&get_data);
        UNITTEST_NE(get_data, (ark::Tensor *)nullptr);
        UNITTEST_EQ(get_data->shape, data->shape);
    }

    return ark::unittest::SUCCESS;
}

ark::unittest::State test_opargs()
{
    ark::OpArgs args;
    UNITTEST_EQ(args.get_args().size(), 0UL);

    ark::Model m;
    auto tns = m.tensor({1, 2, 3, 4}, ark::FP32);

    args.put(ark::OpArg((int)1));
    args.put(ark::OpArg((long long int)2));
    args.put(ark::OpArg((uint64_t)3));
    args.put(ark::OpArg((bool)true));
    args.put(ark::OpArg((float)4));
    args.put(ark::OpArg(ark::Dims(1, 2, 3, 4)));
    args.put(ark::OpArg(tns));

    UNITTEST_EQ(args.get_args().size(), (size_t)7);

    int get_int = 0;
    args.get(&get_int, 0);
    UNITTEST_EQ(get_int, (int)1);

    long long int get_int64 = 0;
    args.get(&get_int64, 1);
    UNITTEST_EQ(get_int64, (long long int)2);

    uint64_t get_uint64 = 0;
    args.get(&get_uint64, 2);
    UNITTEST_EQ(get_uint64, (uint64_t)3);

    bool get_bool = false;
    args.get(&get_bool, 3);
    UNITTEST_EQ(get_bool, true);

    float get_float = 0;
    args.get(&get_float, 4);
    UNITTEST_EQ(get_float, (float)4);

    ark::Dims get_dims;
    args.get(&get_dims, 5);
    UNITTEST_EQ(get_dims, ark::Dims(1, 2, 3, 4));

    ark::Tensor *get_tns = nullptr;
    args.get(&get_tns, 6);
    UNITTEST_NE(get_tns, (ark::Tensor *)nullptr);
    UNITTEST_EQ(get_tns->shape, tns->shape);

    return ark::unittest::SUCCESS;
}

int main()
{
    ark::init();
    UNITTEST(test_oparg);
    UNITTEST(test_opargs);
    return 0;
}
