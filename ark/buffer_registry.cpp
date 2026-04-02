// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "buffer_registry.hpp"

#include "gpu/gpu_logging.hpp"

namespace ark {

BufferRegistry &BufferRegistry::get_instance() {
    static BufferRegistry instance;
    return instance;
}

void BufferRegistry::set(size_t id, void *data, int device_id,
                         bool is_external) {
    if (data != nullptr && device_id < 0) {
        gpuPointerAttributes attr;
        GLOG(gpuPointerGetAttributes(&attr, data));
        device_id = attr.device;
    }
    buffers_[id] =
        std::make_shared<BufferRegistry::Info>(data, device_id, is_external);
}

std::shared_ptr<BufferRegistry::Info> BufferRegistry::get(size_t id) const {
    auto it = buffers_.find(id);
    if (it != buffers_.end()) {
        return it->second;
    }
    return nullptr;
}

}  // namespace ark
