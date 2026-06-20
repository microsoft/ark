// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "ark/model.hpp"
#include "unittest/unittest_utils.h"

ark::unittest::State test_kv_cache_slot_construct() {
    ark::Model m;
    ark::Tensor cache = m.placeholder({4, 2, 3}, ark::FP16);
    ark::Tensor token = m.tensor({2, 3}, ark::FP16);
    ark::Tensor position = m.placeholder({1}, ark::INT32);

    ark::Tensor slot = m.kv_cache_slot(cache, token, position);

    UNITTEST_EQ(slot.shape(), ark::Dims(2, 3));
    UNITTEST_TRUE(slot.data_type() == ark::FP16);
    return ark::unittest::SUCCESS;
}

ark::unittest::State test_kv_cache_slot_invalid() {
    {
        ark::Model m;
        ark::Tensor cache = m.placeholder({4, 2, 3}, ark::FP16);
        ark::Tensor token = m.tensor({2, 4}, ark::FP16);
        ark::Tensor position = m.placeholder({1}, ark::INT32);
        UNITTEST_THROW(m.kv_cache_slot(cache, token, position),
                       ark::ModelError);
    }
    {
        ark::Model m;
        ark::Tensor cache = m.placeholder({4, 2, 3}, ark::FP16, {5, 2, 3}, {},
                                          {4, 2, 3});
        ark::Tensor token = m.tensor({2, 3}, ark::FP16);
        ark::Tensor position = m.placeholder({1}, ark::INT32);
        UNITTEST_THROW(m.kv_cache_slot(cache, token, position),
                       ark::ModelError);
    }
    {
        ark::Model m;
        ark::Tensor cache = m.placeholder({4, 2, 3}, ark::FP16);
        ark::Tensor token = m.tensor({2, 3}, ark::FP16);
        ark::Tensor position = m.placeholder({1}, ark::FP32);
        UNITTEST_THROW(m.kv_cache_slot(cache, token, position),
                       ark::ModelError);
    }
    return ark::unittest::SUCCESS;
}

int main() {
    ark::init();
    UNITTEST(test_kv_cache_slot_construct);
    UNITTEST(test_kv_cache_slot_invalid);
    return ark::unittest::SUCCESS;
}
