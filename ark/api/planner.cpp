// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "ark/planner.hpp"

#include "ark/model.hpp"
#include "env.h"
#include "file_io.h"
#include "gpu/gpu_manager.h"
#include "model/model_json.hpp"
#include "model/model_node.hpp"
#include "model/model_op.hpp"

namespace ark {

class DefaultPlanner::Impl {
   public:
    Impl(const Model &model, int gpu_id);

    void install_config_rule(DefaultPlanner::ConfigRule rule);

    std::string plan(bool pretty) const;

   protected:
    friend class DefaultPlanner;

    Model model_;
    int gpu_id_;
    std::vector<DefaultPlanner::ConfigRule> config_rules_;
};

DefaultPlanner::Impl::Impl(const Model &model, int gpu_id)
    : model_(model.compress()), gpu_id_(gpu_id) {}

void DefaultPlanner::Impl::install_config_rule(
    DefaultPlanner::ConfigRule rule) {
    config_rules_.push_back(
        [rule](const std::string &op, const std::string &arch) -> std::string {
            try {
                return rule(op, arch);
            } catch (const std::exception &e) {
                LOG(WARN, "Skipping a config rule due to an error: ", e.what());
                return "";
            }
        });
}

static void check_config_field(const ModelOpRef op, const Json &config,
                               const std::string &field) {
    if (!config.contains(field)) {
        ERR(NotFoundError, "Config field not found: ", field, " in ",
            op->name());
    }
}

std::string DefaultPlanner::Impl::plan(bool pretty) const {
    const auto gpu_info = GpuManager::get_instance(gpu_id_)->info();
    size_t num_sm = gpu_info.num_sm;
    Json task_infos;
    Json processor_groups;
    size_t max_num_warps = 1;
    size_t max_num_processors = 1;
    size_t next_node_id = 0;
    for (const auto &node : model_.nodes()) {
        for (const auto &op : node->ops) {
            if (op->is_virtual()) continue;

            Json task_info;
            task_info["Id"] = next_node_id++;

            Json config;
            if (!config_rules_.empty()) {
                const std::string op_str = op->serialize().dump();
                for (auto &rule : config_rules_) {
                    auto config_str = rule(op_str, gpu_info.arch->name());
                    if (!config_str.empty()) {
                        config = Json::parse(config_str);
                        break;
                    }
                }
            }
            if (config.empty()) {
                config = op->default_config(gpu_info.arch);
            }
            check_config_field(op, config, "NumWarps");
            check_config_field(op, config, "NumTasks");
            check_config_field(op, config, "SramBytes");
            size_t num_warps = config["NumWarps"];
            size_t num_tasks = config["NumTasks"];
            size_t sram_bytes = config["SramBytes"];
            task_info["NumWarps"] = num_warps;
            task_info["SramBytes"] = sram_bytes;

            max_num_warps = std::max(max_num_warps, num_warps);

            task_info["Ops"] = Json::array();
            task_info["Ops"].push_back(op->serialize());
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

    Json plan;
    plan["Rank"] = model_.rank();
    plan["WorldSize"] = model_.world_size();
    plan["NumProcessors"] = max_num_processors;
    plan["NumWarpsPerProcessor"] = max_num_warps;
    plan["TaskInfos"] = task_infos;
    plan["ProcessorGroups"] = processor_groups;

    std::string plan_str;
    if (pretty) {
        plan_str = PlanJson(plan).dump_pretty();
    } else {
        plan_str = plan.dump();
    }
    const auto &tmp = get_env().path_tmp_dir;
    write_file(tmp + "/model_gpu" + std::to_string(gpu_id_) + ".json",
               model_.serialize());
    write_file(tmp + "/plan_gpu" + std::to_string(gpu_id_) + ".json", plan_str);
    return plan_str;
}

DefaultPlanner::DefaultPlanner(const Model &model, int gpu_id)
    : impl_(std::make_unique<Impl>(model, gpu_id)) {}

DefaultPlanner::~DefaultPlanner() = default;

void DefaultPlanner::install_config_rule(DefaultPlanner::ConfigRule rule) {
    impl_->install_config_rule(rule);
}

std::string DefaultPlanner::plan(bool pretty) const {
    return impl_->plan(pretty);
}

}  // namespace ark
