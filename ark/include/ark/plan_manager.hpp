// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef ARK_PLAN_MANAGER_HPP
#define ARK_PLAN_MANAGER_HPP

#include <ark/context_manager.hpp>

namespace ark {

class PlanManager {
   public:
    PlanManager(Model& model, const std::string& plan_context);

    ~PlanManager();

   private:
    size_t model_id_;
    bool stop_sync_;
    std::shared_ptr<ContextManager> context_manager_;
};

}  // namespace ark

#endif  // ARK_PLAN_MANAGER_HPP
