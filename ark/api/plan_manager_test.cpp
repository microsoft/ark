// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "ark/plan_manager.hpp"
#include "ark/planner.hpp"

#include "model/model_json.hpp"
#include "unittest/unittest_utils.h"

ark::unittest::State test_plan_manager() {
    ark::Model model;
    ark::Tensor t0 = model.tensor({1}, ark::FP32);
    ark::Tensor t1 = model.tensor({1}, ark::FP32);
    ark::Tensor t2 = model.add(t0, t1);

    ark::Tensor t3;
    ark::Tensor t4;
    ark::Tensor t5;
    ark::Tensor t6;
    {
        ark::PlanManager pm_0(model, ark::Json({
            {"processor_range", {0, 2}},
            {"warp_range", {0, 4}},
            {"sram_range", {0, 0}},
            {"sync", false}
        }).dump());
        t3 = model.relu(t2);
        t4 = model.sqrt(t3);
    }
    {
        ark::PlanManager pm_0(model, ark::Json({
            {"processor_range", {2, 4}},
            {"warp_range", {0, 4}},
            {"sram_range", {0, 0}}
        }).dump());
        t5 = model.exp(t2);

        ark::PlanManager pm_1(model, ark::Json({
            {"processor_range", {2, 3}}
        }).dump());
        t6 = model.rsqrt(t5);
    }

    UNITTEST_TRUE(model.verify());

    ark::DefaultPlanner planner(model, 0);
    auto plan_str = planner.plan();
    ark::Json plan = ark::Json::parse(plan_str);

    UNITTEST_LOG(plan_str);

    return ark::unittest::SUCCESS;
}

int main() {
    UNITTEST(test_plan_manager);
    return 0;
}
