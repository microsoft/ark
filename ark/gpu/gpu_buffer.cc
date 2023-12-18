// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "gpu/gpu_buffer.h"

namespace ark {
GpuBuffer::GpuBuffer(int gpu_id, const std::shared_ptr<GpuMemory> memory,
                     int id, size_t offset, size_t bytes)
    : gpu_id_(gpu_id),
      memory_(memory),
      id_(id),
      offset_(offset),
      bytes_(bytes) {}

GpuPtr GpuBuffer::ref(size_t offset) const {
    return memory_->ref(offset_ + offset);
}
}  // namespace ark
