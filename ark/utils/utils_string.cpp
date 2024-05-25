// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "utils_string.hpp"

#include "logging.h"

namespace ark {

bool is_pascal(const std::string &str) {
    if (str.empty()) {
        return false;
    }
    if (!std::isupper(str[0])) {
        return false;
    }
    for (size_t i = 1; i < str.size(); ++i) {
        if (!std::isalnum(str[i])) {
            return false;
        }
    }
    return true;
}

std::string pascal_to_snake(const std::string &str) {
    if (!is_pascal(str)) {
        ERR(InvalidUsageError, "given string (", str,
            ") is not in Pascal case");
    }
    std::string ret;
    for (size_t i = 0; i < str.size(); ++i) {
        if (i > 0 && std::isupper(str[i])) {
            ret.push_back('_');
        }
        ret.push_back(std::tolower(str[i]));
    }
    return ret;
}

std::string to_upper(const std::string &str) {
    std::string ret;
    for (size_t i = 0; i < str.size(); ++i) {
        ret.push_back(std::toupper(str[i]));
    }
    return ret;
}

std::string to_lower(const std::string &str) {
    std::string ret;
    for (size_t i = 0; i < str.size(); ++i) {
        ret.push_back(std::tolower(str[i]));
    }
    return ret;
}

}  // namespace ark
