// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef ARK_GPU_KERNEL_H_
#define ARK_GPU_KERNEL_H_

#include <memory>
#include <string>

#include "gpu/gpu_buffer.h"
#include "gpu/gpu_context.h"

namespace ark {

class GpuKernel {
   public:
    GpuKernel(std::shared_ptr<GpuContext> ctx, const std::string& codes,
              const std::array<int, 3>& block_dim,
              const std::array<int, 3>& grid_dim, size_t smem_bytes,
              const std::string& kernel_name,
              std::initializer_list<std::pair<std::shared_ptr<void>, size_t>>
                  args = {});

    void compile();
    void launch(std::shared_ptr<GpuStream> stream);

   protected:
    std::shared_ptr<GpuContext> ctx_;
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

#endif  // ARK_GPU_KERNEL_H_
