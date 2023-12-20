// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "gpu/gpu_kernel_v2.h"

#include <cstring>

#include "gpu/gpu.h"
#include "gpu/gpu_compile.h"
#include "gpu/gpu_logging.h"

namespace ark {

GpuKernelV2::GpuKernelV2(
    std::shared_ptr<GpuManager> manager, const std::string& codes,
    const std::array<int, 3>& block_dim, const std::array<int, 3>& grid_dim,
    size_t smem_bytes, const std::string& kernel_name,
    std::initializer_list<std::pair<std::shared_ptr<void>, size_t>> args)
    : manager_(manager),
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

void GpuKernelV2::compile() {
    int max_reg_per_block = manager_->info().max_registers_per_block;
    int max_reg_per_thread = manager_->info().max_registers_per_thread;
    int max_reg_cnt =
        max_reg_per_block / (block_dim_[0] * block_dim_[1] * block_dim_[2]);
    if (max_reg_cnt >= max_reg_per_thread) {
        max_reg_cnt = max_reg_per_thread - 1;
    }
    bin_ = gpu_compile({codes_}, manager_->info().arch, max_reg_cnt);
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

GpuState GpuKernelV2::launch(std::shared_ptr<GpuStreamV2> stream) {
    if (!this->is_compiled()) {
        ERR(InvalidUsageError, "Kernel is not compiled yet.");
    }
    return this->manager_->launch(function_, grid_dim_, block_dim_, smem_bytes_,
                                  stream, this->params_ptr_.data(), nullptr);
}

}  // namespace ark
