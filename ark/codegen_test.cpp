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

ark::unittest::State test_codegen_external_buffer_offset_rejected() {
    // send_packet kernels receive OFFSET args into registered ARK memory, not
    // external base pointers. Marking those buffers external must fail instead
    // of generating offsets relative to the wrong base.
    ark::Model model(0, 2);
    ark::Tensor tns = model.tensor({1024}, ark::FP16);
    model.send_packet(tns, 1, /*tag=*/0, /*flag=*/1);

    ark::Planner planner(model, 0);
    auto plan = ark::Json::parse(planner.plan(false));
    UNITTEST_TRUE(plan.contains("TaskInfos"));
    UNITTEST_TRUE(plan["TaskInfos"].size() > 0);

    auto offset_buf_ids = collect_offset_buffer_ids(plan);
    auto tensor_buf_ids = collect_tensor_buffer_ids(plan);
    UNITTEST_TRUE(offset_buf_ids.size() > 0);

    auto &buf_reg = ark::BufferRegistry::get_instance();
    for (size_t id : offset_buf_ids) {
        buf_reg.set(id, reinterpret_cast<void *>(0x1), 0, /*is_external=*/true);
    }

    std::set<size_t> extra;
    extra.insert(offset_buf_ids.begin(), offset_buf_ids.end());
    extra.insert(tensor_buf_ids.begin(), tensor_buf_ids.end());

    ark::PlanJson pj(plan);
    UNITTEST_THROW(ark::CodeGenerator(pj, /*buffer_id_to_offset=*/{}, extra),
                   ark::InternalError);

    return ark::unittest::State::SUCCESS;
}

ark::unittest::State test_codegen_normal_offset() {
    // Non-external OFFSET args are resolved through buffer_id_to_offset_.
    ark::Model model(0, 2);
    ark::Tensor tns = model.tensor({1024}, ark::FP16);
    model.all_reduce_packet(tns, 0, 2);

    ark::Planner planner(model, 0);
    auto plan = ark::Json::parse(planner.plan(false));
    UNITTEST_TRUE(plan.contains("TaskInfos"));
    UNITTEST_TRUE(plan["TaskInfos"].size() > 0);

    auto offset_buf_ids = collect_offset_buffer_ids(plan);
    auto tensor_buf_ids = collect_tensor_buffer_ids(plan);
    UNITTEST_TRUE(offset_buf_ids.size() > 0);

    std::map<size_t, size_t> buf_id_to_offset;
    for (size_t id : offset_buf_ids) {
        buf_id_to_offset[id] = 0;
    }
    for (size_t id : tensor_buf_ids) {
        buf_id_to_offset[id] = 0;
    }

    ark::PlanJson pj(plan);
    ark::CodeGenerator codegen(pj, buf_id_to_offset, {});

    std::string code = codegen.code();
    UNITTEST_TRUE(code.size() > 0);

    return ark::unittest::State::SUCCESS;
}

ark::unittest::State test_all_reduce_packet_external_input_is_staged() {
    ark::Model model(0, 2);
    ark::Tensor input = model.placeholder({1024}, ark::FP16, {}, {}, {}, -1,
                                          reinterpret_cast<void *>(0x1));
    model.all_reduce_packet(input, 0, 2);

    size_t placeholder_id = input.ref()->buffer()->id();
    auto placeholder_info =
        ark::BufferRegistry::get_instance().get(placeholder_id);
    UNITTEST_TRUE(placeholder_info && placeholder_info->is_external);

    std::set<size_t> copy_output_ids;
    bool found_copy_from_placeholder = false;
    bool found_fused = false;
    for (auto &node : model.nodes()) {
        auto &op = node->op;
        if (op->is_virtual()) continue;
        if (op->type() == ark::ModelOpT::from_name("Copy")) {
            auto reads = op->read_tensors();
            auto results = op->result_tensors();
            UNITTEST_TRUE(reads.size() > 0);
            UNITTEST_TRUE(results.size() > 0);
            size_t output_id = results[0]->buffer()->id();
            copy_output_ids.insert(output_id);
            if (reads[0]->buffer()->id() == placeholder_id) {
                found_copy_from_placeholder = true;
            }
        } else if (op->type() ==
                   ark::ModelOpT::from_name("AllReducePacketFused")) {
            found_fused = true;
            auto reads = op->read_tensors();
            UNITTEST_TRUE(reads.size() > 0);
            size_t fused_input_id = reads[0]->buffer()->id();
            UNITTEST_TRUE(fused_input_id != placeholder_id);
            UNITTEST_TRUE(copy_output_ids.count(fused_input_id) > 0);
            auto fused_info =
                ark::BufferRegistry::get_instance().get(fused_input_id);
            UNITTEST_FALSE(fused_info && fused_info->is_external);
        }
    }
    UNITTEST_TRUE(found_copy_from_placeholder);
    UNITTEST_TRUE(found_fused);

    return ark::unittest::State::SUCCESS;
}

ark::unittest::State test_all_reduce_packet_internal_input_is_not_staged() {
    ark::Model model(0, 2);
    ark::Tensor input = model.tensor({1024}, ark::FP16);
    model.all_reduce_packet(input, 0, 2);

    size_t input_id = input.ref()->buffer()->id();
    bool found_copy_from_input = false;
    bool found_fused = false;
    for (auto &node : model.nodes()) {
        auto &op = node->op;
        if (op->is_virtual()) continue;
        if (op->type() == ark::ModelOpT::from_name("Copy")) {
            auto reads = op->read_tensors();
            UNITTEST_TRUE(reads.size() > 0);
            if (reads[0]->buffer()->id() == input_id) {
                found_copy_from_input = true;
            }
        } else if (op->type() ==
                   ark::ModelOpT::from_name("AllReducePacketFused")) {
            found_fused = true;
            auto reads = op->read_tensors();
            UNITTEST_TRUE(reads.size() > 0);
            UNITTEST_TRUE(reads[0]->buffer()->id() == input_id);
        }
    }
    UNITTEST_FALSE(found_copy_from_input);
    UNITTEST_TRUE(found_fused);

    return ark::unittest::State::SUCCESS;
}

ark::unittest::State test_codegen_missing_buffer_id() {
    // Use fresh model buffers so external registrations from earlier tests
    // cannot satisfy this OFFSET lookup.
    ark::Model model(0, 2);
    ark::Tensor tns = model.tensor({512}, ark::FP16);
    model.send_packet(tns, 1, /*tag=*/0, /*flag=*/1);

    ark::Planner planner(model, 0);
    auto plan = ark::Json::parse(planner.plan(false));
    auto offset_buf_ids = collect_offset_buffer_ids(plan);
    UNITTEST_TRUE(offset_buf_ids.size() > 0);
    for (size_t id : offset_buf_ids) {
        auto info = ark::BufferRegistry::get_instance().get(id);
        UNITTEST_FALSE(info && info->is_external);
    }

    ark::PlanJson pj(plan);
    UNITTEST_THROW(ark::CodeGenerator(pj, {}, {}), ark::InternalError);

    return ark::unittest::State::SUCCESS;
}

int main() {
    UNITTEST(test_codegen_external_buffer_offset_rejected);
    UNITTEST(test_codegen_normal_offset);
    UNITTEST(test_all_reduce_packet_external_input_is_staged);
    UNITTEST(test_all_reduce_packet_internal_input_is_not_staged);
    UNITTEST(test_codegen_missing_buffer_id);
    return 0;
}
