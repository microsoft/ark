// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "ark/planner.hpp"

#include "ark/model.hpp"
#include "env.h"
#include "file_io.h"
#include "gpu/gpu_manager.hpp"
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
    Json task_infos = Json::array();
    Json processor_groups = Json::array();
    size_t max_processor_id = 1;
    size_t max_warp_id = 1;
    size_t next_task_id = 0;
    bool prev_append_task = false;
    bool first_op = true;

    auto get_context = [&](const ModelNodeRef &node,
                           const std::string &key) -> Json {
        if (node->context.find(key) != node->context.end()) {
            return Json::parse(node->context.at(key));
        }
        return Json();
    };

    for (const auto &node : model_.nodes()) {
        std::string context = "";
        for (const auto &[key, value] : node->context) {
            context += key + "=" + value + ",";
        }
        context += "prev_append_task=" + std::to_string(prev_append_task);
        LOG(INFO, context);

        for (const auto &op : node->ops) {
            if (op->is_virtual()) continue;

            auto ctx_config = get_context(node, "Config");
            Json config;
            if (!ctx_config.empty()) {
                config = ctx_config;
            } else if (!config_rules_.empty()) {
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

            auto ctx_append_task = get_context(node, "AppendTask");
            if (!ctx_append_task.empty() && ctx_append_task.get<bool>() &&
                prev_append_task) {
                auto &task_info = task_infos.back();
                task_info["NumWarps"] =
                    std::max(task_info["NumWarps"].get<size_t>(), num_warps);
                task_info["SramBytes"] =
                    std::max(task_info["SramBytes"].get<size_t>(), sram_bytes);
                task_info["Ops"].push_back(op->serialize());
                task_info["Ops"].back()["Config"] = config;
            } else {
                Json task_info;
                task_info["Id"] = first_op ? next_task_id : ++next_task_id;
                task_info["NumWarps"] = num_warps;
                task_info["SramBytes"] = sram_bytes;
                task_info["Ops"] = Json::array();
                task_info["Ops"].push_back(op->serialize());
                task_info["Ops"][0]["Config"] = config;
                task_infos.push_back(task_info);

                auto ctx_processor_range = get_context(node, "ProcessorRange");
                auto ctx_warp_range = get_context(node, "WarpRange");
                auto ctx_sram_range = get_context(node, "SramRange");

                Json processor_group;
                if (!ctx_processor_range.empty()) {
                    processor_group["ProcessorRange"] = ctx_processor_range;
                    max_processor_id = std::max(
                        max_processor_id, ctx_processor_range[1].get<size_t>());
                } else {
                    size_t num_processors = std::min(num_sm, num_tasks);
                    processor_group["ProcessorRange"] = {0, num_processors};
                    max_processor_id =
                        std::max(max_processor_id, num_processors);
                }

                Json resource_group;
                resource_group["ProcessorRange"] =
                    processor_group["ProcessorRange"];
                if (!ctx_warp_range.empty()) {
                    resource_group["WarpRange"] = ctx_warp_range;
                    max_warp_id =
                        std::max(max_warp_id, ctx_warp_range[1].get<size_t>());
                } else {
                    resource_group["WarpRange"] = {0, num_warps};
                    max_warp_id = std::max(max_warp_id, num_warps);
                }
                if (!ctx_sram_range.empty()) {
                    resource_group["SramRange"] = ctx_sram_range;
                } else {
                    resource_group["SramRange"] = {0, sram_bytes};
                }
                resource_group["TaskGroups"] = {{{"TaskId", task_info["Id"]},
                                                 {"TaskRange", {0, num_tasks}},
                                                 {"Granularity", 1}}};

                processor_group["ResourceGroups"] = Json::array();
                processor_group["ResourceGroups"].push_back(resource_group);
                processor_groups.push_back(processor_group);
            }
            prev_append_task =
                !ctx_append_task.empty() && ctx_append_task.get<bool>();
            first_op = false;
        }
    }

    Json plan;
    plan["Rank"] = model_.rank();
    plan["WorldSize"] = model_.world_size();
    plan["Architecture"] = gpu_info.arch->name();
    plan["NumProcessors"] = max_processor_id;
    plan["NumWarpsPerProcessor"] = max_warp_id;
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
