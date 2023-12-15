// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef ARK_GPU_BUFFER_H_
#define ARK_GPU_BUFFER_H_

#include "gpu/gpu.h"
#include "gpu/gpu_memory.h"

namespace ark {

using GpuPtr = gpuDeviceptr;

class GpuBuffer {
   public:
    GpuBuffer(int gpu_id, const std::shared_ptr<GpuMemory> memory, int id,
              size_t offset, size_t bytes);
    ~GpuBuffer() = default;
    GpuPtr ref(size_t offset = 0) const;
    size_t get_offset() const { return offset_; }
    void set_offset(size_t offset) { offset_ = offset; }
    int get_gpu_id() const { return gpu_id_; }

   private:
    int gpu_id_;
    std::shared_ptr<GpuMemory> memory_;
    int id_;
    size_t offset_;
    size_t bytes_;
};
}  // namespace ark

#endif  // ARK_GPU_BUFFER_H_
