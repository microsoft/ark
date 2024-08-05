// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "model_context_manager.hpp"

#include "model_node.hpp"
#include "unittest/unittest_utils.h"

ark::unittest::State test_model_context_manager() {
    ark::Model model;
    ark::Tensor t0 = model.tensor({1}, ark::FP32);
    ark::Tensor t1 = model.tensor({1}, ark::FP32);
    ark::Tensor t2 = model.add(t0, t1);

    ark::Tensor t3;
    ark::Tensor t4;
    ark::Tensor t5;
    {
        ark::ModelContextManager cm(model);
        cm.add("key0", "val1");
        t3 = model.relu(t2);

        cm.add("key1", "val2");
        t4 = model.sqrt(t3);
    }
    {
        ark::ModelContextManager cm(model);
        cm.add("key0", "val3");
        t5 = model.exp(t2);
    }

    UNITTEST_TRUE(model.verify());

    auto compressed = model.compress();
    UNITTEST_TRUE(compressed.verify());

    auto nodes = compressed.nodes();
    UNITTEST_EQ(nodes.size(), 4);

    UNITTEST_EQ(nodes[0]->context.size(), 0);
    UNITTEST_EQ(nodes[1]->context.size(), 1);
    UNITTEST_EQ(nodes[1]->context.at("key0"), "val1");
    UNITTEST_EQ(nodes[2]->context.size(), 2);
    UNITTEST_EQ(nodes[2]->context.at("key0"), "val1");
    UNITTEST_EQ(nodes[2]->context.at("key1"), "val2");
    UNITTEST_EQ(nodes[3]->context.size(), 1);
    UNITTEST_EQ(nodes[3]->context.at("key0"), "val3");

    return ark::unittest::SUCCESS;
}

int main() {
    UNITTEST(test_model_context_manager);
    return 0;
}
