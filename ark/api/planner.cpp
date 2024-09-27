// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "ark/planner.hpp"

#include "ark/model.hpp"
#include "context_impl.hpp"
#include "env.h"
#include "file_io.h"
#include "gpu/gpu_manager.hpp"
#include "model/model_json.hpp"
#include "model/model_node.hpp"
#include "model/model_op.hpp"
#include "model/model_tensor.hpp"
#include "range.hpp"

namespace ark {

PlannerContext::PlannerContext(Model &model) : Context(model) {
    this->impl_->set("Id", this->id(), ContextType::Overwrite);
}

void PlannerContext::check_range(const std::string &key,
                                 const Range<int> &range) {
    auto prev = this->impl_->get(key);
    if (prev.empty()) {
        // ok
        return;
    }
    auto prev_vec = prev.get<std::vector<int>>();
    if (prev_vec.size() < 2 || prev_vec.size() > 3) {
        ERR(InternalError, "unexpected");
    }
    int prev_step = (prev_vec.size() == 3) ? prev_vec[2] : 1;
    Range<int> prev_range(prev_vec[0], prev_vec[1], prev_step);
    if (!range.is_subset_of(prev_range)) {
        ERR(PlanError, "New ", key, " ", range,
            " is not a subset of the previous range ", prev_range);
    }
}

void PlannerContext::processor_range(int start, int end, int step) {
    check_range("ProcessorRange", {start, end, step});
    if (step == 1) {
        this->impl_->set("ProcessorRange", {start, end},
                         ContextType::Overwrite);
    } else {
        this->impl_->set("ProcessorRange", {start, end, step},
                         ContextType::Overwrite);
    }
}

void PlannerContext::warp_range(int start, int end, int step) {
    check_range("WarpRange", {start, end, step});
    if (step == 1) {
        this->impl_->set("WarpRange", {start, end}, ContextType::Overwrite);
    } else {
        this->impl_->set("WarpRange", {start, end, step},
                         ContextType::Overwrite);
    }
}

void PlannerContext::sram_range(int start, int end, int step) {
    check_range("SramRange", {start, end, step});
    if (step == 1) {
        this->impl_->set("SramRange", {start, end}, ContextType::Overwrite);
    } else {
        this->impl_->set("SramRange", {start, end, step},
                         ContextType::Overwrite);
    }
}

void PlannerContext::sync(bool sync) {
    if (sync) {
        // `true` should not overwrite `false`.
        if (this->impl_->get("Sync") == Json(false)) {
            LOG(WARN, "Ignoring sync(true) while sync(false) is already set");
            return;
        }
        this->impl_->set("Sync", true, ContextType::Immutable);
    } else {
        this->impl_->set("Sync", false, ContextType::Overwrite);
    }
}

void PlannerContext::config(const std::string &config) {
    this->impl_->set("Config", Json::parse(config), ContextType::Extend);
}

class Planner::Impl {
   public:
    Impl(const Model &model, int device_id);

    void install_config_rule(Planner::ConfigRule rule);

    std::string plan(bool pretty) const;

   protected:
    friend class Planner;

    Model model_;
    int device_id_;
    std::vector<Planner::ConfigRule> config_rules_;
};

Planner::Impl::Impl(const Model &model, int device_id)
    : model_(model.compress()), device_id_(device_id) {}

void Planner::Impl::install_config_rule(Planner::ConfigRule rule) {
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
        ERR(PlanError, "Config field not found: ", field, " in ", op->name());
    }
}

std::string Planner::Impl::plan(bool pretty) const {
    const auto gpu_info = GpuManager::get_instance(device_id_)->info();
    size_t num_sm = gpu_info.num_sm;
    Json task_infos = Json::array();
    Json processor_groups = Json::array();
    size_t max_processor_id = 1;
    size_t max_warp_id = 1;
    size_t next_task_id = 0;
    int prev_ctx_id = -1;
    bool first_op = true;

    auto get_context = [&](const ModelNodeRef &node,
                           const std::string &key) -> Json {
        try {
            return node->context.at(key);
        } catch (const Json::out_of_range &e) {
        }
        return Json();
    };

    auto get_latest_context = [&](const ModelNodeRef &node,
                                  const std::string &key) -> Json {
        auto ctx = get_context(node, key);
        if (ctx.empty()) return Json();
        return ctx.back();
    };

    for (const auto &node : model_.nodes()) {
        const auto &op = node->op;
        if (op->is_virtual()) continue;

        auto ctx_config = get_latest_context(node, "Config");

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
        check_config_field(op, config, "SramBytes");
        size_t num_warps = config["NumWarps"];
        size_t sram_bytes = config["SramBytes"];
        size_t max_num_tasks = 0;
        size_t num_tasks;

        auto &result_tensors = op->result_tensors();
        if (!result_tensors.empty() && config.contains("Tile")) {
            std::stringstream ss;
            ss << "Result shape is not divided by tile. Op: "
               << op->serialize().dump();
            auto not_divided_error = ss.str();

            const std::vector<DimType> tile_vec = config["Tile"];
            auto tile = Dims(tile_vec);
            auto &result_shape = result_tensors[0]->padded_shape();
            if (result_shape.ndims() < tile.ndims()) {
                ERR(PlanError, not_divided_error);
            }
            auto tile4 = tile.dims4();
            auto result_shape4 = result_shape.dims4();
            max_num_tasks = 1;
            for (int i = 0; i < tile4.ndims(); i++) {
                if (tile4[i] == 0 || result_shape4[i] % tile4[i] != 0) {
                    ERR(PlanError, not_divided_error);
                }
                max_num_tasks *= result_shape4[i] / tile4[i];
            }
            if (max_num_tasks == 0) ERR(InternalError, "max_num_tasks == 0");
        }
        if (config.contains("NumTasks")) {
            num_tasks = config["NumTasks"];
            if (max_num_tasks > 0 && num_tasks > max_num_tasks) {
                ERR(PlanError, "NumTasks (", num_tasks,
                    ") exceeds the maximum number of tasks calculated from the "
                    "tile (",
                    max_num_tasks, "). Op: ", op->serialize().dump());
            } else if (num_tasks < max_num_tasks) {
                LOG(WARN, "NumTasks (", num_tasks,
                    ") is less than the maximum number of tasks calculated "
                    "from the tile (",
                    max_num_tasks, "). Op: ", op->serialize().dump());
            }
        } else {
            num_tasks = max_num_tasks;
        }
        if (num_tasks == 0 && op->type() != ModelOpT::from_name("Noop")) {
            LOG(WARN,
                "Detected a non-virtual op that does not perform any "
                "computation. If this is unexpected, please check if "
                "the config includes either `NumTasks` or `Tile` "
                "field. Op: ",
                op->serialize().dump());
        }

        size_t granularity = config.value("Granularity", 1);
        auto ctx_id = get_latest_context(node, "Id");
        auto ctx_sync = get_latest_context(node, "Sync");
        int id = ctx_id.empty() ? -1 : ctx_id.get<int>();
        bool sync = ctx_sync.empty() ? true : ctx_sync.get<bool>();
        if (id == prev_ctx_id && !sync) {
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

            auto ctx_processor_range_list = get_context(node, "ProcessorRange");
            auto ctx_warp_range = get_latest_context(node, "WarpRange");
            auto ctx_sram_range = get_latest_context(node, "SramRange");

            Json processor_group;
            Json resource_group;
            bool new_processor_group = true;
            if (ctx_processor_range_list.empty()) {
                size_t num_processors = std::min(num_sm, num_tasks);
                processor_group["ProcessorRange"] = {0, num_processors};
                resource_group["ProcessorRange"] = {0, num_processors};
                max_processor_id = std::max(max_processor_id, num_processors);
            } else if (ctx_processor_range_list.size() == 1 ||
                       (id != prev_ctx_id)) {
                auto &ctx_processor_range = ctx_processor_range_list[0];
                processor_group["ProcessorRange"] = ctx_processor_range;
                resource_group["ProcessorRange"] = ctx_processor_range;
                max_processor_id = std::max(
                    max_processor_id, ctx_processor_range[1].get<size_t>());
            } else {
                new_processor_group = false;
                resource_group["ProcessorRange"] =
                    ctx_processor_range_list.back();
            }

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
                                             {"Granularity", granularity}}};

            if (new_processor_group) {
                processor_group["ResourceGroups"] = Json::array();
                processor_group["ResourceGroups"].push_back(resource_group);
                processor_groups.push_back(processor_group);
            } else {
                processor_groups.back()["ResourceGroups"].push_back(
                    resource_group);
            }
        }
        prev_ctx_id = id;
        first_op = false;
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
    write_file(tmp + "/model_gpu" + std::to_string(device_id_) + ".json",
               model_.serialize());
    write_file(tmp + "/plan_gpu" + std::to_string(device_id_) + ".json",
               plan_str);
    return plan_str;
}

Planner::Planner(const Model &model, int device_id)
    : impl_(std::make_unique<Impl>(model, device_id)) {}

Planner::~Planner() = default;

void Planner::install_config_rule(Planner::ConfigRule rule) {
    impl_->install_config_rule(rule);
}

std::string Planner::plan(bool pretty) const { return impl_->plan(pretty); }

}  // namespace ark
