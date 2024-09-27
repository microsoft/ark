// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef ARK_CONTEXT_IMPL_HPP_
#define ARK_CONTEXT_IMPL_HPP_

#include "ark/context.hpp"
#include "model/model_json.hpp"

namespace ark {

class ModelContextManager;

class Context::Impl {
   public:
    Impl(Model& model);

    Json get(const std::string& key) const;

    void set(const std::string& key, const Json& value_json, ContextType type);

    bool has(const std::string& key) const;

    Json dump() const;

   protected:
    friend class Context;

    std::shared_ptr<ModelContextManager> context_manager_;
    int id_;
};

}  // namespace ark

#endif  // ARK_CONTEXT_IMPL_HPP_
