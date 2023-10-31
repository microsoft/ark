// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef ARK_GPU_MEM_V2_H_
#define ARK_GPU_MEM_V2_H_

#include <memory>
#include <vector>

#include "gpu/gpu_mgr_v2.h"

namespace ark {

class GpuMgrV2;

class GpuMemV2 {
   public:
    explicit GpuMemV2(GpuMgrV2& gpu_mgr, size_t bytes, size_t align);

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

}  // namespace ark

#endif  // ARK_GPU_MEM_V2_H_
