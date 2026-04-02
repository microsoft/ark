// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "context_impl.hpp"
#include "logging.hpp"

namespace ark {

Context::Context(Model& model) : impl_(std::make_shared<Impl>(model)) {}

int Context::id() const { return this->impl_->id_; }

std::string Context::get(const std::string& key) const {
    if (!this->impl_->has(key)) {
        return "";
    }
    return this->impl_->get(key).dump();
}

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

std::string Context::dump() const {
    return this->impl_->dump().dump();
}

}  // namespace ark
