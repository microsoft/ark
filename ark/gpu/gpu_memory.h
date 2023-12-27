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
    ~GpuMemory() = default;
    void resize(size_t bytes, bool expose = false);
    void resize(const mscclpp::RegisteredMemory& remote_memory);
    size_t bytes() const;

    template <typename T = void>
    T* ref(size_t offset = 0) const {
        return reinterpret_cast<T*>((size_t)this->ref_impl(offset));
    }

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
    void memset(int value, size_t offset, size_t bytes);
    void memset_d32(int value, size_t offset, size_t nelems);
    void memcpy_from(const void* src, size_t offset, size_t bytes,
                     bool from_device = false);
    void memcpy_to(void* dst, size_t offset, size_t bytes);

   private:
    friend class GpuManager;
    GpuMemory(const GpuManager& manager, size_t bytes, size_t align,
              bool expose = false);
    GpuMemory(GpuManager& manager,
              const mscclpp::RegisteredMemory& remote_memory);

    class Impl;
    std::shared_ptr<Impl> pimpl_;

    void* ref_impl(size_t offset = 0) const;
    void to_host(void* dst, size_t bytes, bool async) const;
    void from_host(const void* src, size_t bytes, bool async);
};

class GpuHostMemory {
   public:
    ~GpuHostMemory();
    GpuHostMemory(const GpuHostMemory&) = delete;
    GpuHostMemory& operator=(const GpuHostMemory&) = delete;

    template <typename T>
    T* ref() const {
        return reinterpret_cast<T*>(ptr_);
    }

   private:
    friend class GpuManager;
    GpuHostMemory(const GpuManager& manager, size_t bytes, unsigned int flags);

    void* ptr_;
};

}  // namespace ark

#endif  // ARK_GPU_MEMORY_H_
