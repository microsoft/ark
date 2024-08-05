// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "context_impl.hpp"

#include "logging.hpp"
#include "model/model_context_manager.hpp"
#include "model/model_graph_impl.hpp"

namespace ark {

Context::Impl::Impl(Model& model)
    : context_manager_(std::make_shared<ModelContextManager>(model)) {
    static int next_id = 0;
    id_ = next_id++;
}

Json Context::Impl::get(const std::string& key) const {
    return context_manager_->get(key);
}

void Context::Impl::set(const std::string& key, const Json& value_json,
                        ContextType type) {
    if (type == ContextType::Overwrite) {
        context_manager_->set(key, value_json);
    } else if (type == ContextType::Extend) {
        auto ctx = context_manager_->get(key);
        if (ctx.empty()) {
            context_manager_->set(key, value_json);
        } else if (!ctx.is_object() || !value_json.is_object()) {
            ERR(InvalidUsageError,
                "Context value must be a JSON object when type is "
                "ContextTypeExtend. Key: ",
                key, ", old value: ", ctx.dump(),
                ", new value: ", value_json.dump());
        } else {
            for (const auto& [k, v] : value_json.items()) {
                ctx[k] = v;
            }
            context_manager_->set(key, ctx);
        }
    } else if (type == ContextType::Immutable) {
        if (!context_manager_->has(key)) {
            context_manager_->set(key, value_json);
        }
    } else {
        ERR(InvalidUsageError, "Unknown context type");
    }
}

bool Context::Impl::has(const std::string& key) const {
    return context_manager_->has(key);
}

}  // namespace ark
