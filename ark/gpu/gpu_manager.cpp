// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "gpu/gpu_manager.hpp"

#include <unordered_map>

#include "gpu/gpu_logging.hpp"
#include "utils/utils_string.hpp"

namespace ark {

class GpuManager::Impl {
   public:
    Impl(int gpu_id);
    ~Impl() = default;

   private:
    friend class GpuManager;

    int gpu_id_;
    GpuManager::Info info_;

    void launch(gpuFunction kernel, const std::array<int, 3> &grid_dim,
                const std::array<int, 3> &block_dim, int smem_bytes,
                gpuStream stream, void **params, void **extra);
};

GpuManager::Impl::Impl(int gpu_id) : gpu_id_(gpu_id) {
    GLOG(gpuSetDevice(gpu_id));
    GLOG(gpuDeviceGetAttribute(
        &(info_.cc_major), gpuDeviceAttributeComputeCapabilityMajor, gpu_id_));
    GLOG(gpuDeviceGetAttribute(
        &(info_.cc_minor), gpuDeviceAttributeComputeCapabilityMinor, gpu_id_));
    GLOG(gpuDeviceGetAttribute(&(info_.num_sm),
                               gpuDeviceAttributeMultiprocessorCount, gpu_id_));
    GLOG(gpuDeviceGetAttribute(
        &(info_.smem_total), gpuDeviceAttributeMaxSharedMemoryPerMultiprocessor,
        gpu_id_));
    GLOG(gpuDeviceGetAttribute(&(info_.smem_block_total),
                               gpuDeviceAttributeSharedMemPerBlockOptin,
                               gpu_id_));
    GLOG(gpuDeviceGetAttribute(&(info_.clk_rate), gpuDeviceAttributeClockRate,
                               gpu_id_));
    GLOG(gpuDeviceGetAttribute(&(info_.threads_per_warp),
                               gpuDeviceAttributeWarpSize, gpu_id_));
    GLOG(gpuDeviceGetAttribute(&(info_.max_registers_per_block),
                               gpuDeviceAttributeMaxRegistersPerBlock,
                               gpu_id_));
    GLOG(gpuDeviceGetAttribute(&(info_.max_threads_per_block),
                               gpuDeviceAttributeMaxThreadsPerBlock, gpu_id_));
    size_t gmem_free;
    GLOG(gpuMemGetInfo(&gmem_free, &(info_.gmem_total)));
#if defined(ARK_CUDA)
    auto arch_name =
        "CUDA_" + std::to_string(info_.cc_major * 10 + info_.cc_minor);
#elif defined(ARK_ROCM)
    hipDeviceProp_t prop;
    GLOG(hipGetDeviceProperties(&prop, gpu_id));
    // E.g.: "gfx90a:sramecc+:xnack-"
    std::string gcn_arch_name = prop.gcnArchName;
    if (gcn_arch_name.substr(0, 3) != "gfx") {
        ERR(UnsupportedError,
            "unexpected GCN architecture name: ", gcn_arch_name);
    }
    size_t pos_e = gcn_arch_name.find(":");
    if (pos_e == std::string::npos) {
        ERR(UnsupportedError,
            "unexpected GCN architecture name: ", gcn_arch_name);
    }
    // E.g.: "ROCM_90A"
    auto arch_name = "ROCM_" + to_upper(gcn_arch_name.substr(3, pos_e - 3));
#endif
    info_.arch = Arch::from_name(arch_name);
}

void GpuManager::Impl::launch(gpuFunction kernel,
                              const std::array<int, 3> &grid_dim,
                              const std::array<int, 3> &block_dim,
                              int smem_bytes, gpuStream stream, void **params,
                              void **extra) {
    GLOG_DRV(gpuModuleLaunchKernel(
        kernel, grid_dim[0], grid_dim[1], grid_dim[2], block_dim[0],
        block_dim[1], block_dim[2], smem_bytes, stream, params, extra));
}

std::shared_ptr<GpuManager> GpuManager::get_instance(int gpu_id) {
    static std::unordered_map<int, std::weak_ptr<GpuManager>> instances;
    auto it = instances.find(gpu_id);
    if (it == instances.end()) {
        auto instance = std::shared_ptr<GpuManager>(new GpuManager(gpu_id));
        instances[gpu_id] = instance;
        return instance;
    } else {
        auto instance = it->second.lock();
        if (instance) {
            return instance;
        } else {
            auto instance = std::shared_ptr<GpuManager>(new GpuManager(gpu_id));
            instances[gpu_id] = instance;
            return instance;
        }
    }
}

GpuManager::GpuManager(int gpu_id) : pimpl_(std::make_shared<Impl>(gpu_id)) {}

std::shared_ptr<GpuMemory> GpuManager::malloc(size_t bytes, size_t align,
                                              bool expose) {
    return std::shared_ptr<GpuMemory>(
        new GpuMemory(*this, bytes, align, expose));
}

std::shared_ptr<GpuHostMemory> GpuManager::malloc_host(size_t bytes,
                                                       unsigned int flags) {
    return std::shared_ptr<GpuHostMemory>(
        new GpuHostMemory(*this, bytes, flags));
}

std::shared_ptr<GpuEvent> GpuManager::create_event(bool disable_timing) const {
    return std::shared_ptr<GpuEvent>(
        new GpuEvent(pimpl_->gpu_id_, disable_timing));
}

std::shared_ptr<GpuStream> GpuManager::create_stream() const {
    return std::shared_ptr<GpuStream>(new GpuStream());
}

const GpuManager::Info &GpuManager::info() const { return pimpl_->info_; }

void GpuManager::set_current() const { GLOG(gpuSetDevice(pimpl_->gpu_id_)); }

void GpuManager::launch(gpuFunction function,
                        const std::array<int, 3> &grid_dim,
                        const std::array<int, 3> &block_dim, int smem_bytes,
                        gpuStream stream, void **params, void **extra) const {
    this->set_current();
    pimpl_->launch(function, grid_dim, block_dim, smem_bytes, stream, params,
                   extra);
}

}  // namespace ark
