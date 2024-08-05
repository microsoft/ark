// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef ARK_MODEL_CONTEXT_MANAGER_HPP_
#define ARK_MODEL_CONTEXT_MANAGER_HPP_

#include <map>

#include "ark/model.hpp"
#include "model_graph_impl.hpp"
#include "model_json.hpp"

namespace ark {

class ModelContextManager {
   public:
    ModelContextManager(Model& model);

    ~ModelContextManager();

    void add(const std::string& key, const Json& value);

    bool has(const std::string& key) const;

    Json get(const std::string& key) const;

   private:
    std::shared_ptr<ModelGraphContextStack> context_stack_;
    std::vector<std::string> keys_;
};

}  // namespace ark

#endif  // ARK_MODEL_CONTEXT_MANAGER_HPP_
