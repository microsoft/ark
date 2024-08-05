// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "model_context_manager.hpp"

namespace ark {

ModelContextManager::ModelContextManager(Model& model)
    : context_stack_(model.impl_->context_stack_) {}

ModelContextManager::~ModelContextManager() {
    for (auto it = keys_.rbegin(); it != keys_.rend(); ++it) {
        context_stack_->pop(*it);
    }
}

void ModelContextManager::add(const std::string& key, const Json& value) {
    context_stack_->push(key, value);
    keys_.push_back(key);
}

bool ModelContextManager::has(const std::string& key) const {
    return context_stack_->has(key);
}

Json ModelContextManager::get(const std::string& key) const {
    return context_stack_->get(key);
}

}  // namespace ark
