// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "gpu/gpu_kernel_v2.h"

#include "gpu/gpu.h"
#include "gpu/gpu_compile.h"
#include "gpu/gpu_logging.h"

namespace ark {

class GpuKernelV2::Impl {
   public:
    Impl(std::shared_ptr<GpuMgrV2> gpu_mgr, const std::string& code,
         const std::array<int, 3>& block_dim,
         const std::array<int, 3>& grid_dim, int smem_bytes,
         const std::string& kernel_name);

    void compile();

   private:
    std::shared_ptr<GpuMgrV2> gpu_mgr_;
    std::string code_;
    std::array<int, 3> block_dim_;
    std::array<int, 3> grid_dim_;
    int smem_bytes_;
    std::string kernel_name_;
    std::string bin_;
    gpuModule module_;
    gpuFunction function_;
};

GpuKernelV2::Impl::Impl(std::shared_ptr<GpuMgrV2> gpu_mgr,
                        const std::string& code,
                        const std::array<int, 3>& block_dim,
                        const std::array<int, 3>& grid_dim, int smem_bytes,
                        const std::string& kernel_name)
    : gpu_mgr_(gpu_mgr),
      code_(code),
      block_dim_(block_dim),
      grid_dim_(grid_dim),
      smem_bytes_(smem_bytes),
      kernel_name_(kernel_name) {}

void GpuKernelV2::Impl::compile() {
    int max_reg_per_block = gpu_mgr_->info().max_registers_per_block;
    int max_reg_per_thread = gpu_mgr_->info().max_registers_per_thread;
    int max_reg_cnt =
        max_reg_per_block / (block_dim_[0] * block_dim_[1] * block_dim_[2]);
    if (max_reg_cnt >= max_reg_per_thread) {
        max_reg_cnt = max_reg_per_thread - 1;
    }
    bin_ = gpu_compile({code_}, gpu_mgr_->info().arch, max_reg_cnt);
    GLOG(gpuModuleLoadData(&module_, bin_.c_str()));
    GLOG(gpuModuleGetFunction(&function_, module_, kernel_name_.c_str()));

    int static_smem_size_bytes;
    GLOG(gpuFuncGetAttribute(&static_smem_size_bytes,
                             gpuFuncAttributeSharedSizeBytes, function_));
    int dynamic_smem_size_bytes = smem_bytes_ - static_smem_size_bytes;
    GLOG(gpuFuncSetAttribute(function_,
                             gpuFuncAttributeMaxDynamicSharedSizeBytes,
                             dynamic_smem_size_bytes));
}

GpuKernelV2::GpuKernelV2(std::shared_ptr<GpuMgrV2> gpu_mgr,
                         const std::string& code,
                         const std::array<int, 3>& block_dim,
                         const std::array<int, 3>& grid_dim, int smem_bytes,
                         const std::string& kernel_name)
    : pimpl_(std::make_shared<Impl>(gpu_mgr, code, block_dim, grid_dim,
                                    smem_bytes, kernel_name)) {}

void GpuKernelV2::compile() { pimpl_->compile(); }

}  // namespace ark
