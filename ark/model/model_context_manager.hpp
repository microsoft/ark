// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef ARK_MODEL_CONTEXT_MANAGER_HPP_
#define ARK_MODEL_CONTEXT_MANAGER_HPP_

#include <map>

#include "ark/model.hpp"
#include "model_json.hpp"

namespace ark {

class ModelContextManager {
   public:
    ModelContextManager(Model& model,
                        const std::map<std::string, Json>& context_map);

   private:
    class Impl;
    std::shared_ptr<Impl> impl_;
};

}  // namespace ark

#endif  // ARK_MODEL_CONTEXT_MANAGER_HPP_
