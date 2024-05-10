// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef ARK_UTILS_STRING_HPP_
#define ARK_UTILS_STRING_HPP_

#include <string>

namespace ark {

bool is_pascal(const std::string &str);

std::string pascal_to_snake(const std::string &str);

std::string to_upper(const std::string &str);

std::string to_lower(const std::string &str);

}  // namespace ark

#endif  // ARK_UTILS_STRING_HPP_
