// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "ark/plan_manager.hpp"

#include "logging.h"
#include "model/model_json.hpp"
#include "model/model_graph_impl.hpp"

namespace ark {

class PlanManagerState {
   public:
    PlanManagerState() : sync(true) {}
    bool sync;
};

static std::map<size_t, PlanManagerState> gPlanManagerStates;

PlanManager::PlanManager(Model& model, const std::string& plan_context) : model_id_(model.id()), stop_sync_(false) {
    auto ctx = Json::parse(plan_context);
    if (!ctx.is_object()) {
        ERR(ModelError, "plan context must be a JSON object");
    }
    if (gPlanManagerStates.find(model_id_) == gPlanManagerStates.end()) {
        gPlanManagerStates.emplace(model_id_, PlanManagerState());
    }
    auto& state = gPlanManagerStates[model_id_];
    bool async = !state.sync;
    std::map<std::string, std::string> context_map;
    for (const auto& [key, value] : ctx.items()) {
        if (key == "sync") {
            if (!value.is_boolean()) {
                ERR(ModelError, "sync must be a boolean");
            }
            if (state.sync && !value.get<bool>()) {
                stop_sync_ = true;
                state.sync = false;
                context_map["AppendTask"] = "true";
            } else if (!state.sync) {
                context_map["AppendTask"] = "true";
            }
        } else if (key == "processor_range") {
            if (!value.is_array()) {
                ERR(ModelError, "processor_range must be an array");
            }
            if (async) {
                LOG(WARN, "Ignoring processor_range under sync=false context");
                continue;
            }
            context_map["ProcessorRange"] = value.dump();
        } else if (key == "warp_range") {
            if (!value.is_array()) {
                ERR(ModelError, "warp_range must be an array");
            }
            if (async) {
                LOG(WARN, "Ignoring warp_range under sync=false context");
                continue;
            }
            context_map["WarpRange"] = value.dump();
        } else if (key == "sram_range") {
            if (!value.is_array()) {
                ERR(ModelError, "sram_range must be an array");
            }
            if (async) {
                LOG(WARN, "Ignoring sram_range under sync=false context");
                continue;
            }
            context_map["SramRange"] = value.dump();
        } else if (key == "config") {
            if (!value.is_object()) {
                ERR(ModelError, "config must be an object");
            }
            auto cfg = model.impl_->get_context("Config");
            if (cfg.empty()) {
                context_map["Config"] = value.dump();
            } else {
                auto cfg_obj = Json::parse(cfg);
                for (const auto& [k, v] : value.items()) {
                    cfg_obj[k] = v;
                }
                context_map["Config"] = cfg_obj.dump();
            }
        } else {
            LOG(WARN, "Ignoring unknown plan context key: ", key);
        }
    }
    context_manager_ = std::make_shared<ContextManager>(model, context_map);
}

PlanManager::~PlanManager() {
    if (stop_sync_) {
        gPlanManagerStates[model_id_].sync = true;
    }
}

}  // namespace ark
