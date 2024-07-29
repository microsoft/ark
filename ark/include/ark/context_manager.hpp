// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef ARK_CONTEXT_MANAGER_HPP
#define ARK_CONTEXT_MANAGER_HPP

#include <ark/model.hpp>
#include <map>

namespace ark {

class ContextManager {
   public:
    ContextManager(Model& model,
                   const std::map<std::string, std::string>& context_map);

   private:
    class Impl;
    std::shared_ptr<Impl> impl_;
};

}  // namespace ark

#endif  // ARK_CONTEXT_MANAGER_HPP
