// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "ark/executor.hpp"

#include "ark/dims.hpp"
#include "ark/model.hpp"
#include "codegen.hpp"
#include "model/model_data_type.hpp"
#include "nlohmann/json.hpp"
// #include "planner/planner.hpp"
#include "unittest/unittest_utils.h"

// ark::unittest::State test_executor_scale() {
//     ark::Model m;
//     ark::Tensor input = m.tensor({32}, ark::FP32);
//     ark::Tensor output = m.scale(input, 0.7);

//     auto comp = m.compress();
//     auto serialized = comp.serialize(2);
//     UNITTEST_LOG(serialized);

//     auto comp_json = nlohmann::json::parse(serialized);
//     std::map<size_t, size_t> buf_id_to_bytes;
//     for (auto &tns : comp_json["Tensors"]) {
//         size_t nelems;
//         if (tns.contains("Strides")) {
//             nelems =
//                 ark::Dims(std::vector<ark::DimType>(tns["Strides"])).size();
//         } else {
//             nelems =
//             ark::Dims(std::vector<ark::DimType>(tns["Shape"])).size();
//             UNITTEST_LOG("Shape: ", tns["Shape"].dump(), " ? ", nelems);
//         }
//         size_t bytes =
//             nelems * ark::ModelDataT::from_name(tns["DataType"])->bytes();
//         if (buf_id_to_bytes.find(tns["BufferId"]) != buf_id_to_bytes.end()) {
//             buf_id_to_bytes[tns["BufferId"]] =
//                 std::max(buf_id_to_bytes[tns["BufferId"]], bytes);
//         } else {
//             buf_id_to_bytes[tns["BufferId"]] = bytes;
//         }
//     }

//     // nlohmann::json j;
//     // j["NumProcessors"] = 1;
//     // j["NumWarpsPerProcessor"] = 1;

//     // j["TaskInfos"] = {nlohmann::json()};
//     // j["TaskInfos"][0]["Id"] = 0;
//     // j["TaskInfos"][0]["NumWarps"] = 1;
//     // j["TaskInfos"][0]["SramBytes"] = 0;
//     // j["TaskInfos"][0]["Ops"] = {nlohmann::json()};
//     // j["TaskInfos"][0]["Ops"][0]["Type"] = "Scale";
//     // j["TaskInfos"][0]["Ops"][0]["Name"] = "scale";
//     // j["TaskInfos"][0]["Ops"][0]["IsVirtual"] = false;
//     // j["TaskInfos"][0]["Ops"][0]["ReadTensors"] =
//     {comp_json["Tensors"][0]};
//     // j["TaskInfos"][0]["Ops"][0]["WriteTensors"] =
//     {comp_json["Tensors"][1]};
//     // j["TaskInfos"][0]["Ops"][0]["ResultTensors"] =
//     {comp_json["Tensors"][2]};
//     // j["TaskInfos"][0]["Ops"][0]["Args"] = {
//     //     {"Factor", {"FLOAT", 0.699999988079071}}};
//     // j["TaskInfos"][0]["Ops"][0]["Config"] = nlohmann::json();
//     // j["TaskInfos"][0]["Ops"][0]["Config"]["NumWarps"] = 1;
//     // j["TaskInfos"][0]["Ops"][0]["Config"]["SramBytes"] = 0;
//     // j["TaskInfos"][0]["Ops"][0]["Config"]["Tile"] = {1, 32};

//     // j["ProcessorGroups"] = {nlohmann::json()};
//     // j["ProcessorGroups"][0]["ProcessorRange"] = {0, 1};
//     // j["ProcessorGroups"][0]["ResourceGroups"] = {nlohmann::json()};
//     // j["ProcessorGroups"][0]["ResourceGroups"][0]["ProcessorRange"] = {0,
//     1};
//     // j["ProcessorGroups"][0]["ResourceGroups"][0]["WarpRange"] = {0, 1};
//     // j["ProcessorGroups"][0]["ResourceGroups"][0]["SramRange"] = {0, 1};
//     // j["ProcessorGroups"][0]["ResourceGroups"][0]["TaskGroups"] = {
//     //     nlohmann::json()};
//     //
//     j["ProcessorGroups"][0]["ResourceGroups"][0]["TaskGroups"][0]["TaskId"] =
//     // 0;
//     //
//     j["ProcessorGroups"][0]["ResourceGroups"][0]["TaskGroups"][0]["TaskRange"]
//     // =
//     //     {0, 1};
//     // j["ProcessorGroups"][0]["ResourceGroups"][0]["TaskGroups"][0]
//     //  ["Granularity"] = 1;

//     // auto ctx = ark::GpuContext::get_context(0, 1);
//     // std::map<size_t, std::shared_ptr<ark::GpuBuffer>> buf_id_to_buf;
//     // for (auto &kv : buf_id_to_bytes) {
//     //     auto buf = ctx->allocate_buffer(kv.second, 1);
//     //     buf_id_to_buf[kv.first] = buf;
//     //     UNITTEST_LOG("Allocated buffer ", kv.first, ": offset ",
//     //                  buf->get_offset(), ", bytes ", buf->get_bytes(), " ?
//     ",
//     //                  kv.second);
//     // }
//     // ctx->freeze();

//     // std::map<size_t, size_t> tns_id_to_offset;
//     // for (auto &tns : comp_json["Tensors"]) {
//     //     auto buf = buf_id_to_buf[tns["BufferId"]];
//     //     auto offset = buf->get_offset();
//     //     tns_id_to_offset[tns["Id"]] = offset;
//     //     UNITTEST_LOG("Tensor ", tns["Id"], ": offset ", offset);
//     // }

//     // j["Context"] = nlohmann::json();
//     // j["Context"]["TensorIdToOffset"] = nlohmann::json();
//     // for (auto &kv : tns_id_to_offset) {
//     //     j["Context"]["TensorIdToOffset"][std::to_string(kv.first)] =
//     //     kv.second;
//     // }

//     // UNITTEST_LOG(j.dump(2));

//     ark::Planner planner(m, 0);

//     auto plan = planner.plan(2);
//     UNITTEST_LOG(plan);

//     ark::Executor exe(0, 1, plan, "executor_test");
//     // ark::CodeGenerator codegen(j.dump());
//     // UNITTEST_LOG(codegen.code());
//     exe.compile();
//     exe.launch();
//     exe.run(1);
//     exe.stop();

//     return ark::unittest::SUCCESS;
// }

int main() {
    // UNITTEST(test_executor_scale);
    return 0;
}
