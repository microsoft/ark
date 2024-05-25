// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "gpu_kernel.h"

#include <cassert>
#include <cstring>

#include "gpu.h"
#include "gpu_compile.h"
#include "gpu_logging.h"
#include "gpu_manager.h"

namespace ark {

GpuKernel::GpuKernel(int gpu_id, const std::string& code,
                     const std::array<int, 3>& block_dim,
                     const std::array<int, 3>& grid_dim, size_t smem_bytes,
                     const std::string& kernel_name,
                     std::initializer_list<std::pair<void*, size_t>> args) {
    this->init(gpu_id, code, block_dim, grid_dim, smem_bytes, kernel_name,
               args);
}

void GpuKernel::init(int gpu_id, const std::string& code,
                     const std::array<int, 3>& block_dim,
                     const std::array<int, 3>& grid_dim, size_t smem_bytes,
                     const std::string& kernel_name,
                     std::initializer_list<std::pair<void*, size_t>> args) {
    gpu_manager_ = GpuManager::get_instance(gpu_id);
    code_ = code;
    block_dim_ = block_dim;
    grid_dim_ = grid_dim;
    smem_bytes_ = smem_bytes;
    kernel_name_ = kernel_name;
    params_ptr_.resize(args.size());
    args_.resize(args.size());
    if (kernel_name_.size() == 0) {
        ERR(InvalidUsageError, "Invalid kernel name: ", kernel_name_);
    }
    size_t idx = 0;
    for (auto& pair : args) {
        args_[idx].reset(new uint8_t[pair.second]);
        std::memcpy(args_[idx].get(), &(pair.first), pair.second);
        params_ptr_[idx] = static_cast<void*>(args_[idx].get());
        idx++;
    }
}

void GpuKernel::compile() {
    int max_reg_per_block = gpu_manager_->info().max_registers_per_block;
    int max_reg_per_thread = gpu_manager_->info().max_registers_per_thread;
    int max_reg_cnt =
        max_reg_per_block / (block_dim_[0] * block_dim_[1] * block_dim_[2]);
    if (max_reg_cnt >= max_reg_per_thread) {
        max_reg_cnt = max_reg_per_thread - 1;
    }
    bin_ = gpu_compile({code_}, gpu_manager_->info().arch, max_reg_cnt);
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

void GpuKernel::launch(std::shared_ptr<GpuStream> stream) {
    if (!this->is_compiled()) {
        ERR(InvalidUsageError, "Kernel is not compiled yet.");
    }
    gpu_manager_->launch(function_, grid_dim_, block_dim_, smem_bytes_, stream,
                         params_ptr_.data(), nullptr);
}

gpuDeviceptr GpuKernel::get_global(const std::string& name,
                                   bool ignore_not_found) const {
    gpuDeviceptr ret;
    size_t tmp;
    gpuDrvError err = gpuModuleGetGlobal(&ret, &tmp, module_, name.c_str());
    if ((err == gpuErrorNotFound) && ignore_not_found) {
        return gpuDeviceptr(0);
    } else if (err != gpuDrvSuccess) {
        ERR(GpuError, "Failed to get global variable: ", name);
    }
    return ret;
}

}  // namespace ark
