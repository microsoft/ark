// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "gpu/gpu_manager.h"

#include <unordered_map>

#include "gpu/gpu_logging.h"

namespace ark {
class GpuManager::Impl {
   public:
    Impl(int gpu_id);
    ~Impl() = default;
    Impl(const Impl &) = delete;
    Impl &operator=(const Impl &) = delete;

   private:
    friend class GpuManager;

    int gpu_id_;
    GpuManager::Info info_;
    std::shared_ptr<GpuStream> main_stream_;

    void launch(gpuFunction kernel, const std::array<int, 3> &grid_dim,
                const std::array<int, 3> &block_dim, int smem_bytes,
                std::shared_ptr<GpuStream> stream, void **params, void **extra);

    void memcpy_dtoh_async(void *dst, size_t dst_offset, void *src,
                           size_t src_offset, size_t bytes) const;
    void memcpy_htod_async(void *dst, size_t dst_offset, void *src,
                           size_t src_offset, size_t bytes) const;
    void memcpy_dtod_async(void *dst, size_t dst_offset, void *src,
                           size_t src_offset, size_t bytes) const;
    void memset_async(void *dst, unsigned int val, size_t bytes) const;
    void memset_d32_async(void *dst, unsigned int val, size_t num) const;
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
    info_.arch = "cuda_" + std::to_string(info_.cc_major * 10 + info_.cc_minor);
#elif defined(ARK_ROCM)
    hipDeviceProp_t prop;
    GLOG(hipGetDeviceProperties(&prop, gpu_id));
    // E.g.: "gfx90a:sramecc+:xnack-"
    std::string gcn_arch_name = prop.gcnArchName;
    if (gcn_arch_name.substr(0, 3) != "gfx") {
        ERR(ExecutorError, "unexpected GCN architecture name: ", gcn_arch_name);
    }
    size_t pos_e = gcn_arch_name.find(":");
    if (pos_e == std::string::npos) {
        ERR(ExecutorError, "unexpected GCN architecture name: ", gcn_arch_name);
    }
    // E.g.: "90a"
    info_.arch = "rocm_" + gcn_arch_name.substr(3, pos_e - 3);
#endif
}

void GpuManager::Impl::launch(gpuFunction kernel,
                              const std::array<int, 3> &grid_dim,
                              const std::array<int, 3> &block_dim,
                              int smem_bytes, std::shared_ptr<GpuStream> stream,
                              void **params, void **extra) {
    GLOG_DRV(gpuModuleLaunchKernel(
        kernel, grid_dim[0], grid_dim[1], grid_dim[2], block_dim[0],
        block_dim[1], block_dim[2], smem_bytes, stream->get(), params, extra));
}

void GpuManager::Impl::memcpy_dtoh_async(void *dst, size_t dst_offset,
                                         void *src, size_t src_offset,
                                         size_t bytes) const {
    dst = static_cast<char *>(dst) + dst_offset;
    src = static_cast<char *>(src) + src_offset;
    GLOG(gpuMemcpyAsync(dst, src, bytes, gpuMemcpyDeviceToHost,
                        main_stream_->get()));
}

void GpuManager::Impl::memcpy_htod_async(void *dst, size_t dst_offset,
                                         void *src, size_t src_offset,
                                         size_t bytes) const {
    dst = static_cast<char *>(dst) + dst_offset;
    src = static_cast<char *>(src) + src_offset;
    GLOG(gpuMemcpyAsync(dst, src, bytes, gpuMemcpyHostToDevice,
                        main_stream_->get()));
}

void GpuManager::Impl::memcpy_dtod_async(void *dst, size_t dst_offset,
                                         void *src, size_t src_offset,
                                         size_t bytes) const {
    dst = static_cast<char *>(dst) + dst_offset;
    src = static_cast<char *>(src) + src_offset;
    GLOG(gpuMemcpyAsync(dst, src, bytes, gpuMemcpyDeviceToDevice,
                        main_stream_->get()));
}

void GpuManager::Impl::memset_async(void *dst, unsigned int val,
                                    size_t bytes) const {
    GLOG(gpuMemsetAsync(dst, val, bytes, main_stream_->get()));
}

void GpuManager::Impl::memset_d32_async(void *dst, unsigned int val,
                                        size_t nelems) const {
    GLOG_DRV(
        gpuMemsetD32Async((gpuDeviceptr)dst, val, nelems, main_stream_->get()));
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

GpuManager::GpuManager(int gpu_id) : pimpl_(std::make_shared<Impl>(gpu_id)) {
    this->pimpl_->main_stream_ =
        std::shared_ptr<GpuStream>(new GpuStream(*this));
}

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

std::shared_ptr<GpuEvent> GpuManager::create_event(bool disable_timing) {
    return std::shared_ptr<GpuEvent>(new GpuEvent(*this, disable_timing));
}

std::shared_ptr<GpuStream> GpuManager::create_stream() {
    return std::shared_ptr<GpuStream>(new GpuStream(*this));
}

int GpuManager::get_gpu_id() const { return pimpl_->gpu_id_; }

const GpuManager::Info &GpuManager::info() const { return pimpl_->info_; }

void GpuManager::set_current() const { GLOG(gpuSetDevice(pimpl_->gpu_id_)); }

void GpuManager::memset(void *dst, unsigned int val, size_t bytes,
                        bool async) const {
    this->set_current();
    pimpl_->memset_async(dst, val, bytes);
    if (!async) {
        this->sync();
    }
}

void GpuManager::memcpy_htod(void *dst, size_t dst_offset, void *src,
                             size_t src_offset, size_t bytes,
                             bool async) const {
    this->set_current();
    pimpl_->memcpy_htod_async(dst, dst_offset, src, src_offset, bytes);
    if (!async) {
        this->sync();
    }
}

void GpuManager::memcpy_dtoh(void *dst, size_t dst_offset, void *src,
                             size_t src_offset, size_t bytes,
                             bool async) const {
    this->set_current();
    pimpl_->memcpy_dtoh_async(dst, dst_offset, src, src_offset, bytes);
    if (!async) {
        this->sync();
    }
}

void GpuManager::memcpy_dtod(void *dst, size_t dst_offset, void *src,
                             size_t src_offset, size_t bytes,
                             bool async) const {
    this->set_current();
    pimpl_->memcpy_dtod_async(dst, dst_offset, src, src_offset, bytes);
    if (!async) {
        this->sync();
    }
}

void GpuManager::launch(gpuFunction function,
                        const std::array<int, 3> &grid_dim,
                        const std::array<int, 3> &block_dim, int smem_bytes,
                        std::shared_ptr<GpuStream> stream, void **params,
                        void **extra) const {
    this->set_current();
    pimpl_->launch(function, grid_dim, block_dim, smem_bytes, stream, params,
                   extra);
}

void GpuManager::sync() const { pimpl_->main_stream_->sync(); }

}  // namespace ark
