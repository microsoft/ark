// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef ARK_MODEL_DATA_TYPE_HPP_
#define ARK_MODEL_DATA_TYPE_HPP_

#include <memory>
#include <string>

#include "named_type.hpp"

namespace ark {

class ModelDataT;
using ModelDataType = std::shared_ptr<ModelDataT>;

class ModelDataT : public NamedT {
   public:
    ModelDataT(const std::string &type_name, const std::string &type_str,
               size_t bytes)
        : NamedT(type_name), type_str_(type_str), bytes_(bytes) {}

    ModelDataT(const ModelDataT &) = default;

    const std::string &type_str() const;

    size_t bytes() const;

   private:
    std::string type_str_;
    size_t bytes_;
};

using ModelDataType = std::shared_ptr<ModelDataT>;

}  // namespace ark

#endif  // ARK_MODEL_DATA_TYPE_HPP_
