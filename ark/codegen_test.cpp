// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "codegen.hpp"

#include <map>
#include <nlohmann/json.hpp>
#include <set>

#include "ark/model.hpp"
#include "ark/planner.hpp"
#include "buffer_registry.hpp"
#include "model/model_buffer.hpp"
#include "model/model_node.hpp"
#include "model/model_op.hpp"
#include "model/model_op_arg.hpp"
#include "model/model_tensor.hpp"
#include "unittest/unittest_utils.h"

// Collect all buffer IDs referenced by OFFSET args in a plan's TaskInfos.
static std::set<size_t> collect_offset_buffer_ids(const ark::Json &plan) {
    std::set<size_t> ids;
    for (auto &ti : plan.at("TaskInfos")) {
        for (auto &op_json : ti.at("Ops")) {
            auto op = ark::ModelOp::deserialize(op_json);
            auto args = op->impl_args(op_json.at("Config"));
            for (auto &arg : args) {
                if (arg.type_name() == "OFFSET") {
                    ids.insert(arg.value<ark::ModelOffset>().buffer_id());
                }
            }
        }
    }
    return ids;
}

// Collect all buffer IDs referenced by TENSOR args in a plan's TaskInfos.
static std::set<size_t> collect_tensor_buffer_ids(const ark::Json &plan) {
    std::set<size_t> ids;
    for (auto &ti : plan.at("TaskInfos")) {
        for (auto &op_json : ti.at("Ops")) {
            auto op = ark::ModelOp::deserialize(op_json);
            auto args = op->impl_args(op_json.at("Config"));
            for (auto &arg : args) {
                if (arg.type_name() == "TENSOR") {
                    ids.insert(
                        arg.value<ark::ModelTensorRef>()->buffer()->id());
                }
            }
        }
    }
    return ids;
}

// Test 1: CodeGenerator exercises the external-buffer OFFSET path
// (codegen.cpp line 319: `ss_desc << moff.value();`).
ark::unittest::State test_codegen_external_buffer_offset() {
    // Build a 2-rank model with a send_packet op on rank 0.
    // send_packet's impl_args are two OFFSET args whose buffer IDs we will
    // register as external in BufferRegistry before constructing CodeGenerator.
    ark::Model model(0, 2);
    ark::Tensor tns = model.tensor({1024}, ark::FP16);
    model.send_packet(tns, 1, /*tag=*/0, /*flag=*/1);

    // Plan on GPU 0.
    ark::Planner planner(model, 0);
    auto plan = ark::Json::parse(planner.plan(false));

    // Verify the plan has TaskInfos.
    UNITTEST_TRUE(plan.contains("TaskInfos"));
    UNITTEST_TRUE(plan["TaskInfos"].size() > 0);

    // Collect OFFSET and TENSOR buffer IDs from the plan.
    auto offset_buf_ids = collect_offset_buffer_ids(plan);
    auto tensor_buf_ids = collect_tensor_buffer_ids(plan);
    UNITTEST_TRUE(offset_buf_ids.size() > 0);

    // Register every OFFSET buffer as external in BufferRegistry.
    // Use a dummy non-null pointer; CodeGenerator only checks is_external,
    // it does not dereference the pointer.
    auto &buf_reg = ark::BufferRegistry::get_instance();
    for (size_t id : offset_buf_ids) {
        buf_reg.set(id, reinterpret_cast<void *>(0x1), 0, /*is_external=*/true);
    }

    // All referenced buffer IDs go into extra_buffer_ids (external).
    std::set<size_t> extra;
    extra.insert(offset_buf_ids.begin(), offset_buf_ids.end());
    extra.insert(tensor_buf_ids.begin(), tensor_buf_ids.end());

    // Construct CodeGenerator — exercises the external OFFSET path.
    ark::PlanJson pj(plan);
    ark::CodeGenerator codegen(pj, /*buffer_id_to_offset=*/{}, extra);

    // Verify non-empty generated code.
    std::string code = codegen.code();
    UNITTEST_TRUE(code.size() > 0);

    return ark::unittest::State::SUCCESS;
}

// Test 2: CodeGenerator exercises the normal (non-external) OFFSET path
// (codegen.cpp lines 320-325: buffer_id_to_offset_ lookup).
// Also exercises Model::all_reduce_packet which covers the new
// `input = this->copy(input)` line in ops_all_reduce.cpp:57.
ark::unittest::State test_codegen_normal_offset() {
    // Build a 2-rank model using all_reduce_packet (exercises ops_all_reduce.cpp
    // line 57: `input = this->copy(input)`).
    ark::Model model(0, 2);
    ark::Tensor tns = model.tensor({1024}, ark::FP16);
    model.all_reduce_packet(tns, 0, 2);

    // Plan on GPU 0.
    ark::Planner planner(model, 0);
    auto plan = ark::Json::parse(planner.plan(false));
    UNITTEST_TRUE(plan.contains("TaskInfos"));
    UNITTEST_TRUE(plan["TaskInfos"].size() > 0);

    // Collect ALL buffer IDs (OFFSET + TENSOR) from the plan.
    auto offset_buf_ids = collect_offset_buffer_ids(plan);
    auto tensor_buf_ids = collect_tensor_buffer_ids(plan);
    UNITTEST_TRUE(offset_buf_ids.size() > 0);

    // Put all buffer IDs in buffer_id_to_offset_ with offset 0.
    // Do NOT register them as external in BufferRegistry.
    std::map<size_t, size_t> buf_id_to_offset;
    for (size_t id : offset_buf_ids) {
        buf_id_to_offset[id] = 0;
    }
    for (size_t id : tensor_buf_ids) {
        buf_id_to_offset[id] = 0;
    }

    // Construct CodeGenerator — exercises the normal OFFSET path
    // (buffer_id_to_offset_ lookup, lines 320-325 of codegen.cpp).
    ark::PlanJson pj(plan);
    ark::CodeGenerator codegen(pj, buf_id_to_offset, {});

    std::string code = codegen.code();
    UNITTEST_TRUE(code.size() > 0);

    return ark::unittest::State::SUCCESS;
}

// Test 3: CodeGenerator throws InternalError when an OFFSET arg's buffer ID
// is neither external nor in buffer_id_to_offset (codegen.cpp line 323).
ark::unittest::State test_codegen_missing_buffer_id() {
    // Build a fresh model so its buffer IDs are new (not in BufferRegistry
    // from test 1).
    ark::Model model(0, 2);
    ark::Tensor tns = model.tensor({512}, ark::FP16);
    model.send_packet(tns, 1, /*tag=*/0, /*flag=*/1);

    ark::Planner planner(model, 0);
    auto plan = ark::Json::parse(planner.plan(false));

    // Do NOT register any buffer in BufferRegistry.
    // Do NOT populate buffer_id_to_offset.
    // CodeGenerator should throw InternalError on the OFFSET lookup.
    ark::PlanJson pj(plan);
    UNITTEST_THROW(ark::CodeGenerator(pj, {}, {}), ark::InternalError);

    return ark::unittest::State::SUCCESS;
}

int main() {
    UNITTEST(test_codegen_external_buffer_offset);
    UNITTEST(test_codegen_normal_offset);
    UNITTEST(test_codegen_missing_buffer_id);
    return 0;
}
