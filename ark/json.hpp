// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef ARK_JSON_HPP_
#define ARK_JSON_HPP_

#include <nlohmann/json.hpp>

namespace ark {

using json = ::nlohmann::json;

using ordered_json = ::nlohmann::ordered_json;

}  // namespace ark

#endif  // ARK_JSON_HPP_
