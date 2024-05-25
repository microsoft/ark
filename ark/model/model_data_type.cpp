// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "model_data_type.hpp"

namespace ark {

const std::string &ModelDataT::type_str() const { return type_str_; }

size_t ModelDataT::bytes() const { return bytes_; }

}  // namespace ark
