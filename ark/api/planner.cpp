// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "ark/planner.hpp"

#include "ark/model.hpp"
#include "gpu/gpu_manager.h"
#include "model/model_json.hpp"
#include "model/model_node.hpp"
#include "model/model_op.hpp"

namespace ark {

class DefaultPlanner::Impl {
   public:
    Impl(const Model &model, int gpu_id);

   protected:
    friend class DefaultPlanner;

    Json plan_;
};

DefaultPlanner::Impl::Impl(const Model &model, int gpu_id) {
    const auto &gpu_info = GpuManager::get_instance(gpu_id)->info();
    size_t num_sm = gpu_info.num_sm;
    Json task_infos;
    Json processor_groups;
    size_t max_num_warps = 1;
    size_t max_num_processors = 1;
    size_t next_node_id = 0;
    const auto &compressed = model.compress();
    for (const auto &node : compressed.nodes()) {
        for (const auto &op : node->ops) {
            if (op->is_virtual()) continue;

            Json task_info;
            task_info["Id"] = next_node_id++;

            const auto &config = op->default_config(gpu_info.arch);
            size_t num_warps = config["NumWarps"];
            size_t num_tasks = config["NumTasks"];
            size_t sram_bytes = config["SramBytes"];
            task_info["NumWarps"] = num_warps;
            task_info["SramBytes"] = sram_bytes;

            max_num_warps = std::max(max_num_warps, num_warps);

            task_info["Ops"] = {op->serialize()};
            task_info["Ops"][0]["Config"] = config;
            task_infos.push_back(task_info);

            Json resource_group;
            size_t num_processors = std::min(num_sm, num_tasks);
            max_num_processors = std::max(max_num_processors, num_processors);
            resource_group["ProcessorRange"] = {0, num_processors};
            resource_group["WarpRange"] = {0, num_warps};
            resource_group["SramRange"] = {0, sram_bytes};
            resource_group["TaskGroups"] = {{{"TaskId", task_info["Id"]},
                                             {"TaskRange", {0, num_tasks}},
                                             {"Granularity", 1}}};

            Json processor_group;
            processor_group["ProcessorRange"] = {0, num_processors};
            processor_group["ResourceGroups"] = {resource_group};
            processor_groups.push_back(processor_group);
        }
    }

    plan_["Rank"] = model.rank();
    plan_["WorldSize"] = model.world_size();
    plan_["NumProcessors"] = max_num_processors;
    plan_["NumWarpsPerProcessor"] = max_num_warps;
    plan_["TaskInfos"] = task_infos;
    plan_["ProcessorGroups"] = processor_groups;
}

DefaultPlanner::DefaultPlanner(const Model &model, int gpu_id)
    : impl_(std::make_unique<Impl>(model, gpu_id)) {}

DefaultPlanner::~DefaultPlanner() = default;

std::string DefaultPlanner::plan(bool pretty) const {
    if (pretty) {
        return PlanJson(impl_->plan_).dump_pretty();
    }
    return impl_->plan_.dump();
}

}  // namespace ark
