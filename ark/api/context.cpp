// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "ark/context.hpp"

#include "logging.hpp"
#include "model/model_context_manager.hpp"
#include "model/model_graph_impl.hpp"

namespace ark {

Context::Context(Model& model)
    : context_manager_(std::make_shared<ModelContextManager>(model)) {
    static size_t next_id = 0;
    id_ = next_id++;
}

void Context::set(const std::string& key, const std::string& value,
                  ContextType type) {
    Json value_json;
    try {
        value_json = Json::parse(value);
    } catch (const ::nlohmann::json::parse_error& e) {
        ERR(InvalidUsageError, "Failed to parse context value as JSON: `",
            value, "`");
    }
    if (type == ContextType::ContextTypeOverwrite) {
        context_manager_->set(key, value_json);
    } else if (type == ContextType::ContextTypeExtend) {
        auto ctx = context_manager_->get(key);
        if (ctx.empty()) {
            context_manager_->set(key, value_json);
        } else if (!ctx.is_object() || !value_json.is_object()) {
            ERR(InvalidUsageError,
                "Context value must be a JSON object when type is "
                "ContextTypeExtend. Key: ",
                key, ", old value: ", ctx.dump(), ", new value: ", value);
        } else {
            for (const auto& [k, v] : value_json.items()) {
                ctx[k] = v;
            }
            context_manager_->set(key, ctx);
        }
    } else if (type == ContextType::ContextTypeImmutable) {
        if (!context_manager_->has(key)) {
            context_manager_->set(key, value);
        }
    } else {
        ERR(InvalidUsageError, "Unknown context type");
    }
}

}  // namespace ark
