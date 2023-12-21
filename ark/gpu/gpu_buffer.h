// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef ARK_GPU_BUFFER_H_
#define ARK_GPU_BUFFER_H_

#include "gpu/gpu.h"
#include "gpu/gpu_memory.h"

namespace ark {

class GpuBuffer {
   public:
    GpuBuffer(int gpu_id, const std::shared_ptr<GpuMemory> memory, int id,
              size_t offset, size_t bytes);
    ~GpuBuffer() = default;
    GpuPtr ref(size_t offset = 0) const;
    size_t get_offset() const { return offset_; }
    size_t get_bytes() const { return bytes_; }
    void set_offset(size_t offset) { offset_ = offset; }
    int get_gpu_id() const { return gpu_id_; }
    int get_id() const { return id_; }
    void memset(int value, size_t offset, size_t bytes);
    void memset_d32(int value, size_t offset, size_t nelems);
    void memcpy_from(size_t dst_offset, const void* src, size_t src_offset,
                     size_t bytes);
    void memcpy_to(void* dst, size_t dst_offset, size_t src_offset,
                   size_t bytes);
    void memcpy_from(size_t dst_offset, const GpuBuffer& src, size_t src_offset,
                     size_t bytes);

   private:
    int gpu_id_;
    std::shared_ptr<GpuMemory> memory_;
    int id_;
    size_t offset_;
    size_t bytes_;
};
}  // namespace ark

#endif  // ARK_GPU_BUFFER_H_
