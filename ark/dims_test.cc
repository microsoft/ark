// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "sstream"

#include "include/ark.h"
#include "unittest/unittest_utils.h"

using namespace std;

ark::unittest::State test_dims_basic()
{
    ark::Dims d0{1, 5, 9};
    UNITTEST_TRUE(!d0.is_invalid());
    UNITTEST_TRUE(!d0.is_no_dim());
    UNITTEST_EQ(d0.size(), 45);
    UNITTEST_EQ(d0.ndims(), 3);

    stringstream ss;
    ss << d0;
    UNITTEST_EQ(ss.str(), "<1, 5, 9>");

    ark::Dims d1{1, 5, 9, 1};
    UNITTEST_NE(d0, d1);
    UNITTEST_EQ(d0.size(), d1.size());

    return ark::unittest::SUCCESS;
}

ark::unittest::State test_dims_no_dim()
{
    ark::Dims d0{};
    UNITTEST_TRUE(!d0.is_invalid());
    UNITTEST_TRUE(d0.is_no_dim());
    UNITTEST_EQ(d0.size(), -1);
    UNITTEST_EQ(d0.ndims(), 0);

    stringstream ss;
    ss << d0;
    UNITTEST_EQ(ss.str(), "<>");

    ark::Dims d1;
    UNITTEST_EQ(d0, d1);

    return ark::unittest::SUCCESS;
}

ark::unittest::State test_dims_zero()
{
    ark::Dims d0{0, 10, 0};
    UNITTEST_TRUE(!d0.is_invalid());
    UNITTEST_TRUE(!d0.is_no_dim());
    UNITTEST_EQ(d0.size(), 0);
    UNITTEST_EQ(d0.ndims(), 3);

    stringstream ss;
    ss << d0;
    UNITTEST_EQ(ss.str(), "<0, 10, 0>");

    return ark::unittest::SUCCESS;
}

ark::unittest::State test_dims_from_dims()
{
    ark::Dims d0{1, 2, 3, 4};
    UNITTEST_TRUE(!d0.is_invalid());
    UNITTEST_TRUE(!d0.is_no_dim());

    ark::Dims d1{d0};
    UNITTEST_EQ(d0, d1);

    return ark::unittest::SUCCESS;
}

ark::unittest::State test_dims_from_vector()
{
    vector<ark::DimType> v0{1, 2, 3, 4};
    ark::Dims d0{v0};
    UNITTEST_TRUE(!d0.is_invalid());
    UNITTEST_TRUE(!d0.is_no_dim());

    UNITTEST_EQ(d0.size(), 24);
    UNITTEST_EQ(d0.ndims(), 4);

    UNITTEST_EQ(d0[0], 1);
    UNITTEST_EQ(d0[1], 2);
    UNITTEST_EQ(d0[2], 3);
    UNITTEST_EQ(d0[3], 4);

    return ark::unittest::SUCCESS;
}

int main()
{
    ark::init();
    UNITTEST(test_dims_basic);
    UNITTEST(test_dims_no_dim);
    UNITTEST(test_dims_zero);
    UNITTEST(test_dims_from_dims);
    UNITTEST(test_dims_from_vector);
    return 0;
}
