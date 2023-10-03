// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include <algorithm>

#include "ark.h"

static std::string to_lowercase(const std::string &str) {
    std::string result = str;
    std::transform(result.begin(), result.end(), result.begin(),
                   [](unsigned char c) { return std::tolower(c); });
    return result;
}

namespace ark {

TensorType::TensorType(const std::string &name, int bytes,
                       const std::string &type_str)
    : name_{to_lowercase(name)}, bytes_{bytes}, type_str_{type_str} {}

bool TensorType::operator==(const TensorType &other) const {
    return name_ == other.name();
}

bool TensorType::operator!=(const TensorType &other) const {
    return name_ != other.name();
}

int TensorType::bytes() const { return bytes_; }

const std::string &TensorType::name() const { return name_; }

const std::string &TensorType::type_str() const { return type_str_; }

std::ostream &operator<<(std::ostream &os, const TensorType &type) {
    os << type.name();
    return os;
}

}  // namespace ark
