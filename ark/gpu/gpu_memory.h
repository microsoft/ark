// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef ARK_GPU_MEMORY_H_
#define ARK_GPU_MEMORY_H_

#include <memory>
#include <mscclpp/core.hpp>
#include <vector>

#include "gpu/gpu.h"

namespace ark {

using GpuPtr = gpuDeviceptr;
class GpuManager;

class GpuMemory {
   public:
    GpuMemory(std::shared_ptr<GpuManager> manager, size_t bytes, size_t align,
              bool expose = false);
    GpuMemory(const mscclpp::RegisteredMemory& remote_memory);
    void resize(size_t bytes, bool expose = false);
    void resize(const mscclpp::RegisteredMemory& remote_memory);
    GpuPtr ref(size_t offset = 0) const;
    size_t bytes() const;

    template <typename T>
    void to_host(std::vector<T>& dst, bool async = false) const {
        dst.resize((this->bytes() + sizeof(T) - 1) / sizeof(T));
        this->to_host(dst.data(), async);
    }

    template <typename T>
    void from_host(const std::vector<T>& src, bool async = false) {
        this->from_host(src.data(), src.size() * sizeof(T), async);
    }

    void sync() const;

   private:
    class Impl;
    std::shared_ptr<Impl> pimpl_;

    void to_host(void* dst, bool async) const;
    void from_host(const void* src, size_t bytes, bool async);
};

class GpuHostMemory {
   public:
    GpuHostMemory(std::shared_ptr<GpuManager> manager, size_t bytes,
                  unsigned int flags);
    ~GpuHostMemory();
    template <typename T>
    T* ref(size_t offset = 0) const {
        return reinterpret_cast<T*>(ptr_);
    }

   private:
    void* ptr_;
};

}  // namespace ark

#endif  // ARK_GPU_MEMORY_H_
