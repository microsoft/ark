// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef ARK_GPU_KERNEL_H_
#define ARK_GPU_KERNEL_H_

#include <memory>
#include <string>

#include "gpu_stream.h"

namespace ark {

class GpuManager;

class GpuKernel {
   public:
    GpuKernel(int gpu_id, const std::string& codes,
              const std::array<int, 3>& block_dim,
              const std::array<int, 3>& grid_dim, size_t smem_bytes,
              const std::string& kernel_name,
              std::initializer_list<std::pair<void*, size_t>> args = {});

    void init(int gpu_id, const std::string& codes,
              const std::array<int, 3>& block_dim,
              const std::array<int, 3>& grid_dim, size_t smem_bytes,
              const std::string& kernel_name,
              std::initializer_list<std::pair<void*, size_t>> args = {});
    void compile();
    void launch(std::shared_ptr<GpuStream> stream);

    gpuDeviceptr get_global(const std::string& name,
                            bool ignore_not_found = false) const;
    bool is_compiled() const { return function_ != nullptr; }

   protected:
    std::shared_ptr<GpuManager> gpu_manager_;
    std::string code_;
    std::array<int, 3> block_dim_;
    std::array<int, 3> grid_dim_;
    int smem_bytes_;
    std::string kernel_name_;
    std::string bin_;
    gpuModule module_;
    gpuFunction function_ = nullptr;
    std::vector<void*> params_ptr_;
    std::vector<std::shared_ptr<uint8_t[]>> args_;
};

}  // namespace ark

#endif  // ARK_GPU_KERNEL_H_
