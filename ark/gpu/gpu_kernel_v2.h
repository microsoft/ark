// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef ARK_GPU_KERNEL_V2_H_
#define ARK_GPU_KERNEL_V2_H_

#include <memory>
#include <string>

#include "gpu/gpu_buffer.h"
#include "gpu/gpu_common.h"
#include "gpu/gpu_manager.h"

namespace ark {

class GpuKernelV2 {
   public:
    GpuKernelV2(std::shared_ptr<GpuManager> manager, const std::string& codes,
                const std::array<int, 3>& block_dim,
                const std::array<int, 3>& grid_dim, size_t smem_bytes,
                const std::string& kernel_name,
                std::initializer_list<std::pair<std::shared_ptr<void>, size_t>>
                    args = {});

    void compile();
    GpuState launch(std::shared_ptr<GpuStreamV2> stream);

   protected:
    std::shared_ptr<GpuManager> manager_;
    std::string codes_;
    std::array<int, 3> block_dim_;
    std::array<int, 3> grid_dim_;
    int smem_bytes_;
    std::string kernel_name_;
    std::string bin_;
    gpuModule module_;
    gpuFunction function_ = nullptr;
    std::vector<void*> params_ptr_;
    std::vector<std::shared_ptr<void>> args_;

    bool is_compiled() const { return function_ != nullptr; }
};

}  // namespace ark

#endif  // ARK_GPU_KERNEL_V2_H_
