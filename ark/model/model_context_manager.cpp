// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "model_context_manager.hpp"

#include "model_graph_impl.hpp"

namespace ark {

class ModelContextManager::Impl {
   public:
    Impl(std::shared_ptr<ModelGraphContextStack> context_stack)
        : context_stack_(context_stack) {}

    void add(const std::string& key, const Json& value);

    ~Impl();

   private:
    std::shared_ptr<ModelGraphContextStack> context_stack_;
    std::vector<std::string> keys_;
};

void ModelContextManager::Impl::add(const std::string& key, const Json& value) {
    context_stack_->push(key, value);
    keys_.push_back(key);
}

ModelContextManager::Impl::~Impl() {
    for (auto it = keys_.rbegin(); it != keys_.rend(); ++it) {
        context_stack_->pop(*it);
    }
}

ModelContextManager::ModelContextManager(Model& model)
    : impl_(std::make_shared<Impl>(model.impl_->context_stack_)) {}

ModelContextManager& ModelContextManager::add(const std::string& key,
                                              const Json& value) {
    impl_->add(key, value);
    return *this;
}

}  // namespace ark
