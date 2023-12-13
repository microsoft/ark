// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include <sstream>

#include "include/ark.h"
#include "unittest/unittest_utils.h"

ark::unittest::State test_dims_basic() {
    ark::Dims d0{1, 5, 9};
    UNITTEST_TRUE(!d0.is_invalid());
    UNITTEST_TRUE(!d0.is_no_dim());
    UNITTEST_EQ(d0.size(), 45);
    UNITTEST_EQ(d0.ndims(), 3);

    std::stringstream ss;
    ss << d0;
    UNITTEST_EQ(ss.str(), "<1, 5, 9>");

    ark::Dims d1{1, 5, 9, 1};
    UNITTEST_NE(d0, d1);
    UNITTEST_EQ(d0.size(), d1.size());

    ark::Dims d2{2, 3, 4};
    d2[1] = -1;
    UNITTEST_TRUE(d2.is_invalid());

    // -1 in the middle is not allowed
    auto lambda0 = []() { ark::Dims dims{2, -1, 5}; };
    UNITTEST_THROW(lambda0(), ark::InvalidUsageError);

    // constructing from invalid dims is not allowed
    auto lambda1 = []() {
        ark::Dims dims{2, 3, 4};
        dims[1] = -1;
        ark::Dims new_dims{dims};
    };
    UNITTEST_THROW(lambda1(), ark::InvalidUsageError);

    // too long
    auto lambda2 = []() {
        std::vector<ark::DimType> v0{1, 2, 3, 4, 5};
        ark::Dims d0{v0};
    };
    UNITTEST_THROW(lambda2(), ark::InvalidUsageError);

    // -1 in the middle is not allowed
    auto lambda3 = []() {
        std::vector<ark::DimType> v0{5, -1, 2, 3};
        ark::Dims d0{v0};
    };
    UNITTEST_THROW(lambda3(), ark::InvalidUsageError);

    auto lambda4 = []() {
        std::vector<ark::DimType> vec{1, 2, 3, -3};
        ark::Dims dim{vec};
    };
    UNITTEST_THROW(lambda4(), ark::InvalidUsageError);

    return ark::unittest::SUCCESS;
}

ark::unittest::State test_dims_no_dim() {
    ark::Dims d0{};
    UNITTEST_TRUE(!d0.is_invalid());
    UNITTEST_TRUE(d0.is_no_dim());
    UNITTEST_EQ(d0.size(), -1);
    UNITTEST_EQ(d0.ndims(), 0);

    std::stringstream ss;
    ss << d0;
    UNITTEST_EQ(ss.str(), "<>");

    ark::Dims d1;
    UNITTEST_EQ(d0, d1);

    return ark::unittest::SUCCESS;
}

ark::unittest::State test_dims_zero() {
    ark::Dims d0{0, 10, 0};
    UNITTEST_TRUE(!d0.is_invalid());
    UNITTEST_TRUE(!d0.is_no_dim());
    UNITTEST_EQ(d0.size(), 0);
    UNITTEST_EQ(d0.ndims(), 3);

    std::stringstream ss;
    ss << d0;
    UNITTEST_EQ(ss.str(), "<0, 10, 0>");

    return ark::unittest::SUCCESS;
}

ark::unittest::State test_dims_from_dims() {
    ark::Dims d0{1, 2, 3, 4};
    UNITTEST_TRUE(!d0.is_invalid());
    UNITTEST_TRUE(!d0.is_no_dim());

    ark::Dims d1{d0};
    UNITTEST_EQ(d0, d1);

    return ark::unittest::SUCCESS;
}

ark::unittest::State test_dims_from_vector() {
    std::vector<ark::DimType> v0{1, 2, 3, 4};
    ark::Dims d0{v0};
    UNITTEST_TRUE(!d0.is_invalid());
    UNITTEST_TRUE(!d0.is_no_dim());

    UNITTEST_EQ(d0.size(), 24);
    UNITTEST_EQ(d0.ndims(), 4);

    UNITTEST_EQ(d0[0], 1);
    UNITTEST_EQ(d0[1], 2);
    UNITTEST_EQ(d0[2], 3);
    UNITTEST_EQ(d0[3], 4);

    UNITTEST_THROW(d0[4], ark::InvalidUsageError);

    return ark::unittest::SUCCESS;
}

ark::unittest::State test_dims_neg_index() {
    ark::Dims d0{10, 20, 30, 40};

    UNITTEST_EQ(d0[-0], 10);
    UNITTEST_EQ(d0[-1], 40);
    UNITTEST_EQ(d0[-2], 30);
    UNITTEST_EQ(d0[-3], 20);
    UNITTEST_EQ(d0[-4], 10);

    UNITTEST_THROW(d0[-5], ark::InvalidUsageError);

    return ark::unittest::SUCCESS;
}

ark::unittest::State test_dims_erase() {
    ark::Dims d0{10, 20, 30, 40};

    UNITTEST_EQ(d0[0], 10);
    UNITTEST_EQ(d0[1], 20);
    UNITTEST_EQ(d0[2], 30);
    UNITTEST_EQ(d0[3], 40);
    UNITTEST_EQ(d0.ndims(), 4);

    ark::DimType ret = d0.erase(1);
    UNITTEST_EQ(ret, 20);

    UNITTEST_EQ(d0[0], 10);
    UNITTEST_EQ(d0[1], 30);
    UNITTEST_EQ(d0[2], 40);
    UNITTEST_EQ(d0.ndims(), 3);
    UNITTEST_EQ(d0.size(), 12000);
    UNITTEST_TRUE(!d0.is_invalid());
    UNITTEST_TRUE(!d0.is_no_dim());

    UNITTEST_THROW(d0.erase(3), ark::InvalidUsageError);

    ret = d0.erase(-1);
    UNITTEST_EQ(ret, 40);

    return ark::unittest::SUCCESS;
}

ark::unittest::State test_dims_ostream() {
    ark::Dims d0{10, 20, 30, 40};
    std::stringstream ss;
    ss << d0;
    UNITTEST_EQ(ss.str(), "<10, 20, 30, 40>");

    d0[1] = -1;
    UNITTEST_TRUE(d0.is_invalid());
    ss.str("");

    auto lamda = [&]() { ss << d0; };
    UNITTEST_THROW(lamda(), ark::InvalidUsageError);

    return ark::unittest::SUCCESS;
}

int main() {
    ark::init();
    UNITTEST(test_dims_basic);
    UNITTEST(test_dims_no_dim);
    UNITTEST(test_dims_zero);
    UNITTEST(test_dims_from_dims);
    UNITTEST(test_dims_from_vector);
    UNITTEST(test_dims_neg_index);
    UNITTEST(test_dims_erase);
    UNITTEST(test_dims_ostream);
    return 0;
}
