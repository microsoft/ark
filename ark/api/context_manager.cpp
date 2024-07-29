// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "ark/context_manager.hpp"

#include "model/model_graph_impl.hpp"

namespace ark {

class ContextManager::Impl {
   public:
    Impl(std::shared_ptr<ModelGraphContextStack> context_stack,
         const std::map<std::string, std::string>& context_map);

    ~Impl();

   private:
    std::shared_ptr<ModelGraphContextStack> context_stack_;
    std::vector<std::string> keys_;
};

ContextManager::Impl::Impl(
    std::shared_ptr<ModelGraphContextStack> context_stack,
    const std::map<std::string, std::string>& context_map)
    : context_stack_(context_stack) {
    for (const auto& [key, value] : context_map) {
        context_stack_->push(key, value);
        keys_.push_back(key);
    }
}

ContextManager::Impl::~Impl() {
    for (auto it = keys_.rbegin(); it != keys_.rend(); ++it) {
        context_stack_->pop(*it);
    }
}

ContextManager::ContextManager(
    Model& model, const std::map<std::string, std::string>& context_map)
    : impl_(std::make_shared<Impl>(model.impl_->context_stack_, context_map)) {}

}  // namespace ark
