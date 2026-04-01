// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "ark/planner.hpp"

#include "model/model_node.hpp"
#include "unittest/unittest_utils.h"

ark::unittest::State test_planner_context_processor_range() {
    {
        ark::Model model;
        ark::Tensor t0 = model.tensor({1}, ark::FP32);
        ark::Tensor t1 = model.tensor({1}, ark::FP32);

        // node 0
        ark::Tensor t2 = model.add(t0, t1);

        ark::Tensor t3;
        ark::Tensor t4;
        ark::Tensor t5;
        int ctx_id_1;
        int ctx_id_2;
        {
            // node 1
            ark::PlannerContext ctx(model);
            ctx_id_1 = ctx.id();
            ctx.processor_range(0, 4);
            t3 = model.relu(t2);

            UNITTEST_EQ(ctx.get("ProcessorRange"),
                        ark::Json({ctx.id(), {0, 4}}).dump());

            // node 2
            ctx.processor_range(2, 4);
            t4 = model.sqrt(t3);

            UNITTEST_EQ(ctx.get("ProcessorRange"),
                        ark::Json({ctx.id(), {2, 4}}).dump());

            // Invalid usage: range (0, 4) is out of previous range (2, 4)
            UNITTEST_THROW(ctx.processor_range(0, 4), ark::PlanError);
        }
        {
            // node 3
            ark::PlannerContext ctx(model);
            ctx_id_2 = ctx.id();
            ctx.processor_range(2, 6, 2);
            t5 = model.exp(t2);

            UNITTEST_EQ(ctx.get("ProcessorRange"),
                        ark::Json({ctx.id(), {2, 6, 2}}).dump());
        }

        UNITTEST_TRUE(model.verify());

        auto compressed = model.compress();
        UNITTEST_TRUE(compressed.verify());

        auto nodes = compressed.nodes();
        UNITTEST_EQ(nodes.size(), 4);

        UNITTEST_EQ(nodes[0]->context.size(), 0);
        UNITTEST_GE(nodes[1]->context.size(), 1);
        UNITTEST_EQ(nodes[1]->context.at("ProcessorRange"),
                    ark::Json({{ctx_id_1, {0, 4}}}));
        UNITTEST_GE(nodes[2]->context.size(), 1);
        UNITTEST_EQ(nodes[2]->context.at("ProcessorRange"),
                    ark::Json({{ctx_id_1, {0, 4}}, {ctx_id_1, {2, 4}}}));
        UNITTEST_GE(nodes[3]->context.size(), 1);
        UNITTEST_EQ(nodes[3]->context.at("ProcessorRange"),
                    ark::Json({{ctx_id_2, {2, 6, 2}}}));
    }
    {
        ark::Model model;
        ark::Tensor t0 = model.tensor({1}, ark::FP32);
        ark::Tensor t1 = model.tensor({1}, ark::FP32);

        ark::PlannerContext ctx(model);
        ctx.processor_range(0, 10);

        std::vector<ark::Tensor> tensors;
        std::vector<int> subctx_ids;
        for (size_t i = 0; i < 5; ++i) {
            ark::PlannerContext subctx(model);
            subctx_ids.push_back(subctx.id());
            subctx.processor_range(0 * i, 2 * i);
            auto t = model.add(t0, t1);
            tensors.push_back(t);

            UNITTEST_EQ(ctx.get("ProcessorRange"),
                        ark::Json({subctx.id(), {0 * (int)i, 2 * (int)i}}).dump());
        }

        UNITTEST_TRUE(model.verify());

        auto compressed = model.compress();
        UNITTEST_TRUE(compressed.verify());

        auto nodes = compressed.nodes();
        UNITTEST_EQ(nodes.size(), 5);

        for (size_t i = 0; i < 5; ++i) {
            UNITTEST_GE(nodes[i]->context.size(), 1);
            // Each node has the outer ctx + subctx stacked
            auto pr = nodes[i]->context.at("ProcessorRange");
            UNITTEST_EQ(pr.back(),
                        ark::Json({subctx_ids[i], {0 * (int)i, 2 * (int)i}}));
        }
    }

    return ark::unittest::SUCCESS;
}

ark::unittest::State test_planner_context_warp_range() {
    ark::Model model;
    ark::Tensor t0 = model.tensor({1}, ark::FP32);
    ark::Tensor t1 = model.tensor({1}, ark::FP32);

    // node 0
    ark::Tensor t2 = model.add(t0, t1);

    ark::Tensor t3;
    ark::Tensor t4;
    ark::Tensor t5;
    int wctx_id_1;
    int wctx_id_2;
    {
        // node 1
        ark::PlannerContext ctx(model);
        wctx_id_1 = ctx.id();
        ctx.warp_range(0, 4);
        t3 = model.relu(t2);

        UNITTEST_EQ(ctx.get("WarpRange"),
                    ark::Json({ctx.id(), {0, 4}}).dump());

        // node 2
        ctx.warp_range(2, 4);
        t4 = model.sqrt(t3);

        UNITTEST_EQ(ctx.get("WarpRange"),
                    ark::Json({ctx.id(), {2, 4}}).dump());

        // Invalid usage: range (0, 4) is out of previous range (2, 4)
        UNITTEST_THROW(ctx.warp_range(0, 4), ark::PlanError);
    }
    {
        // node 3
        ark::PlannerContext ctx(model);
        wctx_id_2 = ctx.id();
        ctx.warp_range(2, 6, 2);
        t5 = model.exp(t2);

        UNITTEST_EQ(ctx.get("WarpRange"),
                    ark::Json({ctx.id(), {2, 6, 2}}).dump());
    }

    UNITTEST_TRUE(model.verify());

    auto compressed = model.compress();
    UNITTEST_TRUE(compressed.verify());

    auto nodes = compressed.nodes();
    UNITTEST_EQ(nodes.size(), 4);

    UNITTEST_EQ(nodes[0]->context.size(), 0);
    UNITTEST_GE(nodes[1]->context.size(), 1);
    UNITTEST_EQ(nodes[1]->context.at("WarpRange"),
                ark::Json({{wctx_id_1, {0, 4}}}));
    UNITTEST_GE(nodes[2]->context.size(), 1);
    UNITTEST_EQ(nodes[2]->context.at("WarpRange"),
                ark::Json({{wctx_id_1, {0, 4}}, {wctx_id_1, {2, 4}}}));
    UNITTEST_GE(nodes[3]->context.size(), 1);
    UNITTEST_EQ(nodes[3]->context.at("WarpRange"),
                ark::Json({{wctx_id_2, {2, 6, 2}}}));

    return ark::unittest::SUCCESS;
}

ark::unittest::State test_planner_context_sram_range() {
    ark::Model model;
    ark::Tensor t0 = model.tensor({1}, ark::FP32);
    ark::Tensor t1 = model.tensor({1}, ark::FP32);

    // node 0
    ark::Tensor t2 = model.add(t0, t1);

    ark::Tensor t3;
    ark::Tensor t4;
    ark::Tensor t5;
    int sctx_id_1;
    int sctx_id_2;
    {
        // node 1
        ark::PlannerContext ctx(model);
        sctx_id_1 = ctx.id();
        ctx.sram_range(0, 4);
        t3 = model.relu(t2);

        UNITTEST_EQ(ctx.get("SramRange"),
                    ark::Json({ctx.id(), {0, 4}}).dump());

        // node 2
        ctx.sram_range(2, 4);
        t4 = model.sqrt(t3);

        UNITTEST_EQ(ctx.get("SramRange"),
                    ark::Json({ctx.id(), {2, 4}}).dump());

        // Invalid usage: range (0, 4) is out of previous range (2, 4)
        UNITTEST_THROW(ctx.sram_range(0, 4), ark::PlanError);
    }
    {
        // node 3
        ark::PlannerContext ctx(model);
        sctx_id_2 = ctx.id();
        ctx.sram_range(2, 6, 2);
        t5 = model.exp(t2);

        UNITTEST_EQ(ctx.get("SramRange"),
                    ark::Json({ctx.id(), {2, 6, 2}}).dump());
    }

    UNITTEST_TRUE(model.verify());

    auto compressed = model.compress();
    UNITTEST_TRUE(compressed.verify());

    auto nodes = compressed.nodes();
    UNITTEST_EQ(nodes.size(), 4);

    UNITTEST_EQ(nodes[0]->context.size(), 0);
    UNITTEST_GE(nodes[1]->context.size(), 1);
    UNITTEST_EQ(nodes[1]->context.at("SramRange"),
                ark::Json({{sctx_id_1, {0, 4}}}));
    UNITTEST_GE(nodes[2]->context.size(), 1);
    UNITTEST_EQ(nodes[2]->context.at("SramRange"),
                ark::Json({{sctx_id_1, {0, 4}}, {sctx_id_1, {2, 4}}}));
    UNITTEST_GE(nodes[3]->context.size(), 1);
    UNITTEST_EQ(nodes[3]->context.at("SramRange"),
                ark::Json({{sctx_id_2, {2, 6, 2}}}));

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
    int sync_id_1;
    int sync_id_2;
    {
        // node 1
        ark::PlannerContext ctx(model);
        sync_id_1 = ctx.id();
        ctx.sync(false);
        t3 = model.relu(t2);

        UNITTEST_EQ(ctx.get("Sync"),
                    ark::Json({ctx.id(), false}).dump());

        // node 2
        ctx.sync(true);
        t4 = model.sqrt(t3);

        UNITTEST_EQ(ctx.get("Sync"),
                    ark::Json({ctx.id(), true}).dump());
    }
    {
        // node 3
        ark::PlannerContext ctx(model);
        sync_id_2 = ctx.id();
        ctx.sync(true);
        t5 = model.exp(t2);

        UNITTEST_EQ(ctx.get("Sync"),
                    ark::Json({ctx.id(), true}).dump());
    }

    UNITTEST_TRUE(model.verify());

    auto compressed = model.compress();
    UNITTEST_TRUE(compressed.verify());

    auto nodes = compressed.nodes();
    UNITTEST_EQ(nodes.size(), 4);

    UNITTEST_EQ(nodes[0]->context.size(), 0);
    UNITTEST_GE(nodes[1]->context.size(), 1);
    UNITTEST_EQ(nodes[1]->context.at("Sync"),
                ark::Json({{sync_id_1, true}, {sync_id_1, false}}));
    UNITTEST_GE(nodes[2]->context.size(), 1);
    UNITTEST_EQ(nodes[2]->context.at("Sync"),
                ark::Json({{sync_id_1, true}, {sync_id_1, false}, {sync_id_1, true}}));
    UNITTEST_GE(nodes[3]->context.size(), 1);
    UNITTEST_EQ(nodes[3]->context.at("Sync"),
                ark::Json({{sync_id_2, true}, {sync_id_2, true}}));

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
    int cfg_id_1;
    int cfg_id_2;
    {
        // node 1
        ark::PlannerContext ctx(model);
        cfg_id_1 = ctx.id();
        ctx.config(ark::Json({{"key0", "val1"}}).dump());
        t3 = model.relu(t2);

        UNITTEST_EQ(ctx.get("Config"),
                    ark::Json({ctx.id(), {{"key0", "val1"}}}).dump());

        // node 2
        ctx.config(ark::Json({{"key1", "val2"}}).dump());
        t4 = model.sqrt(t3);

        UNITTEST_EQ(ctx.get("Config"),
                    ark::Json({ctx.id(), {{"key1", "val2"}}}).dump());
    }
    {
        // node 3
        ark::PlannerContext ctx(model);
        cfg_id_2 = ctx.id();
        ctx.config(ark::Json({{"key2", "val3"}}).dump());
        t5 = model.exp(t2);

        UNITTEST_EQ(ctx.get("Config"),
                    ark::Json({ctx.id(), {{"key2", "val3"}}}).dump());
    }

    UNITTEST_TRUE(model.verify());

    auto compressed = model.compress();
    UNITTEST_TRUE(compressed.verify());

    auto nodes = compressed.nodes();
    UNITTEST_EQ(nodes.size(), 4);

    UNITTEST_EQ(nodes[0]->context.size(), 0);
    UNITTEST_GE(nodes[1]->context.size(), 1);
    UNITTEST_EQ(nodes[1]->context.at("Config"),
                ark::Json({{cfg_id_1, {{"key0", "val1"}}}}));
    UNITTEST_GE(nodes[2]->context.size(), 1);
    UNITTEST_EQ(nodes[2]->context.at("Config"),
                ark::Json({{cfg_id_1, {{"key0", "val1"}}}, {cfg_id_1, {{"key1", "val2"}}}}));
    UNITTEST_GE(nodes[3]->context.size(), 1);
    UNITTEST_EQ(nodes[3]->context.at("Config"),
                ark::Json({{cfg_id_2, {{"key2", "val3"}}}}));

    return ark::unittest::SUCCESS;
}

ark::unittest::State test_planner_context_plan() {
    ark::Model model;
    ark::PlannerContext ctx(model);
    ctx.processor_range(0, 2);
    ctx.warp_range(0, 4);
    ctx.sram_range(0, 0);
    ctx.sync(false);
    ark::Json cfg({{"NumWarps", 1},
                   {"SramBytes", 0},
                   {"NumTasks", 1},
                   {"Tile", {1, 64}}});
    ctx.config(cfg.dump());

    ark::Tensor t0 = model.tensor({1024}, ark::FP32);
    ark::Tensor t1 = model.mul(t0, 0.5);
    ark::Tensor t2 = model.add(t0, t1);

    ark::Planner planner(model, 0);
    auto plan = ark::Json::parse(planner.plan(false));

    UNITTEST_EQ(plan["NumProcessors"].get<int>(), 2);
    UNITTEST_EQ(plan["NumWarpsPerProcessor"].get<int>(), 4);
    UNITTEST_EQ(plan["TaskInfos"].size(), 1);
    UNITTEST_EQ(plan["TaskInfos"][0]["NumWarps"], 1);
    UNITTEST_EQ(plan["TaskInfos"][0]["SramBytes"], 0);
    UNITTEST_EQ(plan["TaskInfos"][0]["Ops"].size(), 2);
    UNITTEST_EQ(plan["TaskInfos"][0]["Ops"][0]["Type"].get<std::string>(),
                "ScalarMul");
    UNITTEST_EQ(plan["TaskInfos"][0]["Ops"][0]["Config"], cfg);
    UNITTEST_EQ(plan["TaskInfos"][0]["Ops"][1]["Type"].get<std::string>(),
                "Add");
    UNITTEST_EQ(plan["TaskInfos"][0]["Ops"][1]["Config"], cfg);

    return ark::unittest::SUCCESS;
}

int main() {
    UNITTEST(test_planner_context_processor_range);
    UNITTEST(test_planner_context_warp_range);
    UNITTEST(test_planner_context_sram_range);
    UNITTEST(test_planner_context_sync);
    UNITTEST(test_planner_context_config);
    UNITTEST(test_planner_context_plan);
    return 0;
}
