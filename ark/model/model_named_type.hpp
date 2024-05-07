// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef ARK_MODEL_NAMED_TYPE_HPP_
#define ARK_MODEL_NAMED_TYPE_HPP_

#include <string>

namespace ark {

class ModelNamedT {
   public:
    ModelNamedT(const std::string &type_name) : type_name_(type_name) {}
    ModelNamedT &operator=(const ModelNamedT &) = default;

    const std::string &type_name() const { return type_name_; }

   private:
    std::string type_name_;
};

}  // namespace ark

#endif  // ARK_MODEL_NAMED_TYPE_HPP_
