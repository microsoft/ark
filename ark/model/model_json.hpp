// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef ARK_MODEL_JSON_HPP_
#define ARK_MODEL_JSON_HPP_

#include <nlohmann/json.hpp>

namespace ark {

using Json = ::nlohmann::ordered_json;

class ModelJson : public Json {
   public:
    ModelJson(const Json &json) : Json(json) {}
    std::string dump_pretty(int indent = 0, int indent_step = 2) const;
};

class PlanJson : public Json {
   public:
    PlanJson(const Json &json) : Json(json) {}
    std::string dump_pretty(int indent = 0, int indent_step = 2) const;
};

}  // namespace ark

#endif  // ARK_MODEL_JSON_HPP_
