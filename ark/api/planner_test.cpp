// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "ark/planner.hpp"

#include "model/model_node.hpp"
#include "unittest/unittest_utils.h"

ark::unittest::State test_planner_context_range() {
    ark::Model model;
    ark::Tensor t0 = model.tensor({1}, ark::FP32);
    ark::Tensor t1 = model.tensor({1}, ark::FP32);

    // node 0
    ark::Tensor t2 = model.add(t0, t1);

    ark::Tensor t3;
    ark::Tensor t4;
    ark::Tensor t5;
    {
        // node 1
        ark::PlannerContext ctx(model);
        ctx.processor_range(0, 4);
        t3 = model.relu(t2);

        UNITTEST_EQ(ctx.get("ProcessorRange"), ark::Json({0, 4}).dump());

        // node 2
        ctx.processor_range(2, 4);
        t4 = model.sqrt(t3);

        UNITTEST_EQ(ctx.get("ProcessorRange"), ark::Json({2, 4}).dump());

        // Invalid usage: range (0, 4) is out of previous range (2, 4)
        UNITTEST_THROW(ctx.processor_range(0, 4), ark::PlanError);
    }
    {
        // node 3
        ark::PlannerContext ctx(model);
        ctx.processor_range(2, 6);
        t5 = model.exp(t2);

        UNITTEST_EQ(ctx.get("ProcessorRange"), ark::Json({2, 6}).dump());
    }

    UNITTEST_TRUE(model.verify());

    auto compressed = model.compress();
    UNITTEST_TRUE(compressed.verify());

    auto nodes = compressed.nodes();
    UNITTEST_EQ(nodes.size(), 4);

    UNITTEST_EQ(nodes[0]->context.size(), 0);
    UNITTEST_GE(nodes[1]->context.size(), 1);
    UNITTEST_EQ(nodes[1]->context.at("ProcessorRange"), ark::Json({0, 4}));
    UNITTEST_GE(nodes[2]->context.size(), 1);
    UNITTEST_EQ(nodes[2]->context.at("ProcessorRange"), ark::Json({2, 4}));
    UNITTEST_GE(nodes[3]->context.size(), 1);
    UNITTEST_EQ(nodes[3]->context.at("ProcessorRange"), ark::Json({2, 6}));

    return ark::unittest::SUCCESS;
}

ark::unittest::State test_planner_context_sync() {
    ark::Model model;
    ark::Tensor t0 = model.tensor({1}, ark::FP32);
    ark::Tensor t1 = model.tensor({1}, ark::FP32);

    // node 0
    ark::Tensor t2 = model.add(t0, t1);

    ark::Tensor t3;
    ark::Tensor t4;
    ark::Tensor t5;
    {
        // node 1
        ark::PlannerContext ctx(model);
        ctx.sync(false);
        t3 = model.relu(t2);

        UNITTEST_EQ(ctx.get("Sync"), ark::Json(false).dump());

        // node 2
        ctx.sync(true);  // will be ignored with a warning message
        t4 = model.sqrt(t3);

        UNITTEST_EQ(ctx.get("Sync"), ark::Json(false).dump());
    }
    {
        // node 3
        ark::PlannerContext ctx(model);
        t5 = model.exp(t2);

        UNITTEST_EQ(ctx.get("Sync"), "");
    }

    UNITTEST_TRUE(model.verify());

    auto compressed = model.compress();
    UNITTEST_TRUE(compressed.verify());

    auto nodes = compressed.nodes();
    UNITTEST_EQ(nodes.size(), 4);

    UNITTEST_EQ(nodes[0]->context.size(), 0);
    UNITTEST_GE(nodes[1]->context.size(), 1);
    UNITTEST_EQ(nodes[1]->context.at("Sync"), ark::Json(false));
    UNITTEST_GE(nodes[2]->context.size(), 1);
    UNITTEST_EQ(nodes[2]->context.at("Sync"), ark::Json(false));

    return ark::unittest::SUCCESS;
}

ark::unittest::State test_planner_context_config() {
    ark::Model model;
    ark::Tensor t0 = model.tensor({1}, ark::FP32);
    ark::Tensor t1 = model.tensor({1}, ark::FP32);

    // node 0
    ark::Tensor t2 = model.add(t0, t1);

    ark::Tensor t3;
    ark::Tensor t4;
    ark::Tensor t5;
    {
        // node 1
        ark::PlannerContext ctx(model);
        ctx.config(ark::Json({{"key0", "val1"}}).dump());
        t3 = model.relu(t2);

        UNITTEST_EQ(ctx.get("Config"), ark::Json({{"key0", "val1"}}).dump());

        // node 2
        ctx.config(ark::Json({{"key1", "val2"}}).dump());
        t4 = model.sqrt(t3);

        UNITTEST_EQ(ctx.get("Config"),
                    ark::Json({{"key0", "val1"}, {"key1", "val2"}}).dump());
    }
    {
        // node 3
        ark::PlannerContext ctx(model);
        ctx.config(ark::Json({{"key2", "val3"}}).dump());
        t5 = model.exp(t2);

        UNITTEST_EQ(ctx.get("Config"), ark::Json({{"key2", "val3"}}).dump());
    }

    UNITTEST_TRUE(model.verify());

    auto compressed = model.compress();
    UNITTEST_TRUE(compressed.verify());

    auto nodes = compressed.nodes();
    UNITTEST_EQ(nodes.size(), 4);

    UNITTEST_EQ(nodes[0]->context.size(), 0);
    UNITTEST_GE(nodes[1]->context.size(), 1);
    UNITTEST_EQ(nodes[1]->context.at("Config"), ark::Json({{"key0", "val1"}}));
    UNITTEST_GE(nodes[2]->context.size(), 1);
    UNITTEST_EQ(nodes[2]->context.at("Config"),
                ark::Json({{"key0", "val1"}, {"key1", "val2"}}));
    UNITTEST_GE(nodes[3]->context.size(), 1);
    UNITTEST_EQ(nodes[3]->context.at("Config"), ark::Json({{"key2", "val3"}}));

    return ark::unittest::SUCCESS;
}

int main() {
    UNITTEST(test_planner_context_range);
    UNITTEST(test_planner_context_sync);
    UNITTEST(test_planner_context_config);
    return 0;
}
