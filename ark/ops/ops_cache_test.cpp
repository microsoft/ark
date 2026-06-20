// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "ark/model.hpp"
#include "model/model_node.hpp"
#include "model/model_op.hpp"
#include "unittest/unittest_utils.h"

ark::unittest::State test_kv_cache_slot_construct() {
    ark::Model m;
    ark::Tensor cache = m.placeholder({4, 2, 3}, ark::FP16);
    ark::Tensor token = m.tensor({2, 3}, ark::FP16);
    ark::Tensor position = m.placeholder({1}, ark::INT32);

    ark::Tensor slot = m.kv_cache_slot(cache, token, position);

    UNITTEST_EQ(slot.shape(), ark::Dims(2, 3));
    UNITTEST_TRUE(slot.data_type() == ark::FP16);

    ark::Tensor output = m.tensor({2, 3}, ark::FP16);
    ark::Tensor slot_out = m.kv_cache_slot(cache, token, position, output);
    UNITTEST_EQ(slot_out.shape(), ark::Dims(2, 3));
    UNITTEST_TRUE(slot_out.data_type() == ark::FP16);
    return ark::unittest::SUCCESS;
}

ark::unittest::State test_kv_cache_slot_orders_shared_state() {
    ark::Model m;
    ark::Tensor cache = m.placeholder({4, 2, 3}, ark::FP16);
    ark::Tensor token = m.tensor({2, 3}, ark::FP16);
    ark::Tensor position = m.placeholder({1}, ark::INT32);

    ark::Tensor slot0 = m.kv_cache_slot(cache, token, position);
    ark::Tensor slot1 = m.kv_cache_slot(cache, token, position);

    ark::Model compressed = m.compress();
    auto nodes = compressed.nodes();
    ark::ModelNodeRef slot0_node;
    ark::ModelNodeRef slot1_node;
    for (auto &node : nodes) {
        if (node->op->result_tensors().empty()) {
            continue;
        }
        if (node->op->result_tensors()[0] == slot0.ref()) {
            slot0_node = node;
        } else if (node->op->result_tensors()[0] == slot1.ref()) {
            slot1_node = node;
        }
    }

    UNITTEST_TRUE(slot0_node != nullptr);
    UNITTEST_TRUE(slot1_node != nullptr);
    UNITTEST_TRUE(slot1_node->producers.contains(slot0_node));
    UNITTEST_TRUE(slot0_node->consumers.contains(slot1_node));
    return ark::unittest::SUCCESS;
}

ark::unittest::State test_kv_cache_slot_orders_writer_after_reader() {
    ark::Model m;
    ark::Tensor cache = m.placeholder({4, 2, 3}, ark::FP16);
    ark::Tensor token = m.tensor({2, 3}, ark::FP16);
    ark::Tensor position = m.placeholder({1}, ark::INT32);

    m.kv_cache_slot(cache, token, position);
    ark::Tensor read = m.copy(cache);
    ark::Tensor slot1 = m.kv_cache_slot(cache, token, position);

    ark::Model compressed = m.compress();
    auto nodes = compressed.nodes();
    ark::ModelNodeRef read_node;
    ark::ModelNodeRef slot1_node;
    for (auto &node : nodes) {
        if (node->op->result_tensors().empty()) {
            continue;
        }
        if (node->op->result_tensors()[0] == read.ref()) {
            read_node = node;
        } else if (node->op->result_tensors()[0] == slot1.ref()) {
            slot1_node = node;
        }
    }

    UNITTEST_TRUE(read_node != nullptr);
    UNITTEST_TRUE(slot1_node != nullptr);
    UNITTEST_TRUE(slot1_node->producers.contains(read_node));
    UNITTEST_TRUE(read_node->consumers.contains(slot1_node));
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
        ark::Tensor cache =
            m.placeholder({4, 2, 3}, ark::FP16, {5, 2, 3}, {}, {4, 2, 3});
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
    {
        ark::Model m;
        ark::Tensor cache = m.tensor({4, 2, 3}, ark::FP16);
        ark::Tensor token = m.tensor({2, 3}, ark::FP16);
        ark::Tensor position = m.placeholder({1}, ark::INT32);
        UNITTEST_THROW(m.kv_cache_slot(cache, token, position),
                       ark::ModelError);
    }
    {
        ark::Model m;
        ark::Tensor cache = m.placeholder({4, 2, 3}, ark::FP16);
        ark::Tensor token = m.tensor({2, 3}, ark::FP16);
        ark::Tensor position = m.tensor({1}, ark::INT32);
        UNITTEST_THROW(m.kv_cache_slot(cache, token, position),
                       ark::ModelError);
    }
    return ark::unittest::SUCCESS;
}

int main() {
    ark::init();
    UNITTEST(test_kv_cache_slot_construct);
    UNITTEST(test_kv_cache_slot_orders_shared_state);
    UNITTEST(test_kv_cache_slot_orders_writer_after_reader);
    UNITTEST(test_kv_cache_slot_invalid);
    return ark::unittest::SUCCESS;
}
