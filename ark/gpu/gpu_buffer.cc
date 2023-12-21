// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "gpu/gpu_buffer.h"

#include "gpu/gpu_logging.h"

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
void GpuBuffer::memset(int value, size_t offset, size_t bytes) {
    const size_t& buffer_bytes = this->get_bytes();
    if (buffer_bytes < bytes) {
        ERR(InvalidUsageError,
            "memset requests too many elements. Expected <= ", buffer_bytes,
            ", given ", bytes);
    }
    memory_->memset(value, offset_ + offset, bytes);
}

void GpuBuffer::memset_d32(int value, size_t offset, size_t nelems) {
    this->memset(value, offset, nelems * sizeof(int));
}

void GpuBuffer::memcpy_from(size_t dst_offset, const void* src,
                            size_t src_offset, size_t bytes) {
    void* host_ptr = reinterpret_cast<void*>((size_t)src + src_offset);
    memory_->memcpy_from(host_ptr, offset_ + dst_offset, bytes);
}

void GpuBuffer::memcpy_to(void* dst, size_t dst_offset, size_t src_offset,
                          size_t bytes) {
    void* host_ptr = reinterpret_cast<void*>((size_t)dst + dst_offset);
    memory_->memcpy_to(host_ptr, offset_ + src_offset, bytes);
}

void GpuBuffer::memcpy_from(size_t dst_offset, const GpuBuffer& src,
                            size_t src_offset, size_t bytes) {
    memory_->memcpy_from((void*)src.ref(src_offset), offset_ + dst_offset,
                         bytes);
}

}  // namespace ark
