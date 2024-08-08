// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef ARK_GPU_KERNEL_HPP_
#define ARK_GPU_KERNEL_HPP_

#include <memory>
#include <string>
#include <vector>

#include "gpu_stream.hpp"

namespace ark {

class GpuManager;

class GpuKernel {
   public:
    GpuKernel(int gpu_id, const std::string& codes,
              const std::array<int, 3>& block_dim,
              const std::array<int, 3>& grid_dim, size_t smem_bytes,
              const std::string& kernel_name);

    void init(int gpu_id, const std::string& codes,
              const std::array<int, 3>& block_dim,
              const std::array<int, 3>& grid_dim, size_t smem_bytes,
              const std::string& kernel_name);
    void compile();
    void launch(gpuStream stream, std::vector<void*>& args);

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
};

}  // namespace ark

#endif  // ARK_GPU_KERNEL_HPP_
