// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef ARK_GPU_MEMORY_H_
#define ARK_GPU_MEMORY_H_

#include <memory>
#include <vector>

#include "gpu/gpu.h"

namespace ark {

class GpuManager;

class GpuMemory {
   public:
    ~GpuMemory() = default;
    size_t bytes() const;

    template <typename T = void>
    T* ref(size_t offset = 0) const {
        return reinterpret_cast<T*>((size_t)this->ref_impl(offset));
    }

   private:
    friend class GpuManager;
    GpuMemory(const GpuManager& manager, size_t bytes, size_t align,
              bool expose = false);

    class Impl;
    std::shared_ptr<Impl> pimpl_;

    void* ref_impl(size_t offset = 0) const;
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
