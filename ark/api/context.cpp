// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "context_impl.hpp"
#include "logging.hpp"

namespace ark {

Context::Context(Model& model) : impl_(std::make_shared<Impl>(model)) {}

size_t Context::id() const { return this->impl_->id_; }

void Context::set(const std::string& key, const std::string& value,
                  ContextType type) {
    Json value_json;
    try {
        value_json = Json::parse(value);
    } catch (const ::nlohmann::json::parse_error& e) {
        ERR(InvalidUsageError, "Failed to parse context value as JSON: `",
            value, "`");
    }
    this->impl_->set(key, value_json, type);
}

}  // namespace ark
