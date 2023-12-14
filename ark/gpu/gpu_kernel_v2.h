// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef ARK_GPU_KERNEL_V2_H_
#define ARK_GPU_KERNEL_V2_H_

#include <memory>
#include <string>

#include "gpu/gpu_manager.h"

namespace ark {

class GpuKernelV2 {
   public:
    GpuKernelV2(std::shared_ptr<GpuManager> manager, const std::string& code,
                const std::array<int, 3>& block_dim,
                const std::array<int, 3>& grid_dim, int smem_bytes,
                const std::string& kernel_name);

    void compile();

   private:
    class Impl;
    std::shared_ptr<Impl> pimpl_;
};

}  // namespace ark

#endif  // ARK_GPU_KERNEL_V2_H_
