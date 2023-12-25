// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "gpu/gpu_kernel.h"

#include <cassert>
#include <cstring>

#include "gpu/gpu.h"
#include "gpu/gpu_compile.h"
#include "gpu/gpu_logging.h"

namespace ark {

GpuKernel::GpuKernel(
    std::shared_ptr<GpuContext> ctx, const std::string& codes,
    const std::array<int, 3>& block_dim, const std::array<int, 3>& grid_dim,
    size_t smem_bytes, const std::string& kernel_name,
    std::initializer_list<std::pair<std::shared_ptr<void>, size_t>> args)
    : ctx_(ctx),
      codes_(codes),
      block_dim_(block_dim),
      grid_dim_(grid_dim),
      smem_bytes_(smem_bytes),
      kernel_name_(kernel_name),
      params_ptr_(args.size(), nullptr),
      args_(args.size(), nullptr) {
    if (kernel_name_.size() == 0) {
        ERR(InvalidUsageError, "Invalid kernel name: ", kernel_name_);
    }
    int idx = 0;
    for (auto& pair : args) {
        std::shared_ptr<void> ptr =
            std::shared_ptr<uint8_t[]>(new uint8_t[pair.second]);
        assert(ptr != nullptr);
        if (pair.first != nullptr) {
            std::memcpy(ptr.get(), pair.first.get(), pair.second);
        }
        // make sure the shared_ptr is not released
        this->args_[idx] = ptr;
        this->params_ptr_[idx++] = ptr.get();
    }
}

void GpuKernel::compile() {
    auto manager = ctx_->get_gpu_manager();
    int max_reg_per_block = manager->info().max_registers_per_block;
    int max_reg_per_thread = manager->info().max_registers_per_thread;
    int max_reg_cnt =
        max_reg_per_block / (block_dim_[0] * block_dim_[1] * block_dim_[2]);
    if (max_reg_cnt >= max_reg_per_thread) {
        max_reg_cnt = max_reg_per_thread - 1;
    }
    bin_ = gpu_compile({codes_}, manager->info().arch, max_reg_cnt);
    GLOG_DRV(gpuModuleLoadData(&module_, bin_.c_str()));
    GLOG_DRV(gpuModuleGetFunction(&function_, module_, kernel_name_.c_str()));

    int static_smem_size_bytes;
    GLOG_DRV(gpuFuncGetAttribute(&static_smem_size_bytes,
                                 gpuFuncAttributeSharedSizeBytes, function_));
    int dynamic_smem_size_bytes = smem_bytes_ - static_smem_size_bytes;
    GLOG_DRV(gpuFuncSetAttribute(function_,
                                 gpuFuncAttributeMaxDynamicSharedSizeBytes,
                                 dynamic_smem_size_bytes));
}

GpuState GpuKernel::launch(std::shared_ptr<GpuStream> stream) {
    if (!this->is_compiled()) {
        ERR(InvalidUsageError, "Kernel is not compiled yet.");
    }
    return ctx_->get_gpu_manager()->launch(function_, grid_dim_, block_dim_,
                                           smem_bytes_, stream,
                                           this->params_ptr_.data(), nullptr);
}

}  // namespace ark
