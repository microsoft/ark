// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "external_buffer_registry.hpp"

#include "logging.hpp"

namespace ark {

ExternalBufferRegistry &ExternalBufferRegistry::get_instance() {
    static ExternalBufferRegistry instance;
    return instance;
}

void ExternalBufferRegistry::set(const size_t id, void *data) {
    if (data == nullptr) {
        ERR(InternalError, "data is nullptr.");
    }
    buffers_[id] = data;
}

void *ExternalBufferRegistry::get(const size_t id) const {
    auto it = buffers_.find(id);
    if (it != buffers_.end()) {
        return it->second;
    }
    return nullptr;
}

void ExternalBufferRegistry::clear() { buffers_.clear(); }

}  // namespace ark
