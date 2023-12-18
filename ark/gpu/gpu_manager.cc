// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "gpu/gpu_manager.h"

#include <unordered_map>

#include "gpu/gpu.h"
#include "gpu/gpu_logging.h"

namespace ark {
class GpuManager::Impl {
   public:
    Impl(int gpu_id);
    ~Impl();
    Impl(const Impl &) = delete;
    Impl &operator=(const Impl &) = delete;

   private:
    friend class GpuManager;

    int gpu_id_;
    gpuDevice gpu_dev_;
    gpuCtx gpu_ctx_;
    GpuManager::Info info_;
    std::shared_ptr<GpuStreamV2> main_stream_;
};

GpuManager::Impl::Impl(int gpu_id) : gpu_id_(gpu_id) {
    if (gpuDeviceGet(&gpu_dev_, gpu_id) == gpuErrorNotInitialized) {
        GLOG(gpuInit(0));
        GLOG(gpuDeviceGet(&gpu_dev_, gpu_id));
    }
    GLOG(gpuDevicePrimaryCtxRetain(&gpu_ctx_, gpu_dev_));
    GLOG(gpuCtxSetCurrent(gpu_ctx_));

    GLOG(gpuDeviceGetAttribute(
        &(info_.cc_major), gpuDeviceAttributeComputeCapabilityMajor, gpu_dev_));
    GLOG(gpuDeviceGetAttribute(
        &(info_.cc_minor), gpuDeviceAttributeComputeCapabilityMinor, gpu_dev_));
    GLOG(gpuDeviceGetAttribute(
        &(info_.num_sm), gpuDeviceAttributeMultiprocessorCount, gpu_dev_));
    GLOG(gpuDeviceGetAttribute(
        &(info_.smem_total), gpuDeviceAttributeMaxSharedMemoryPerMultiprocessor,
        gpu_dev_));
    GLOG(gpuDeviceGetAttribute(&(info_.smem_block_total),
                               gpuDeviceAttributeSharedMemPerBlockOptin,
                               gpu_dev_));
    GLOG(gpuDeviceGetAttribute(&(info_.clk_rate), gpuDeviceAttributeClockRate,
                               gpu_dev_));
    GLOG(gpuDeviceGetAttribute(&(info_.threads_per_warp),
                               gpuDeviceAttributeWarpSize, gpu_dev_));
    GLOG(gpuDeviceGetAttribute(&(info_.max_registers_per_block),
                               gpuDeviceAttributeMaxRegistersPerBlock,
                               gpu_dev_));
    GLOG(gpuDeviceGetAttribute(&(info_.max_threads_per_block),
                               gpuDeviceAttributeMaxThreadsPerBlock, gpu_dev_));
    size_t gmem_free;
    GLOG(gpuMemGetInfo(&gmem_free, &(info_.gmem_total)));

    main_stream_ = std::make_shared<GpuStreamV2>();
}

GpuManager::Impl::~Impl() {
    auto e = gpuDevicePrimaryCtxRelease(gpu_dev_);
    if (e != gpuErrorDeinitialized) GLOG(e);
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
    return std::make_shared<GpuMemory>(
        GpuManager::get_instance(pimpl_->gpu_id_), bytes, align, expose);
}

int GpuManager::get_gpu_id() const { return pimpl_->gpu_id_; }

const GpuManager::Info &GpuManager::info() const { return pimpl_->info_; }

void GpuManager::set_current() const {
    GLOG(gpuCtxSetCurrent(pimpl_->gpu_ctx_));
}

void GpuManager::memset_d32_sync(void *dst, unsigned int val,
                                 size_t num) const {
    this->set_current();
    this->memset_d32_async(dst, val, num);
    this->sync();
}

void GpuManager::memcpy_dtoh_async(void *dst, size_t dst_offset, void *src,
                                   size_t src_offset, size_t bytes) const {
    dst = static_cast<char *>(dst) + dst_offset;
    gpuDeviceptr d_src = (gpuDeviceptr)((uint64_t)src + src_offset);
    GLOG(gpuMemcpyDtoHAsync(dst, d_src, bytes, pimpl_->main_stream_->get()));
}

void GpuManager::memcpy_htod_async(void *dst, size_t dst_offset, void *src,
                                   size_t src_offset, size_t bytes) const {
    gpuDeviceptr d_dst = reinterpret_cast<gpuDeviceptr>(
        reinterpret_cast<long long unsigned int>(dst) + dst_offset);
    src = static_cast<char *>(src) + src_offset;
    GLOG(gpuMemcpyHtoDAsync(d_dst, src, bytes, pimpl_->main_stream_->get()));
}

void GpuManager::memcpy_dtod_async(void *dst, size_t dst_offset, void *src,
                                   size_t src_offset, size_t bytes) const {
    gpuDeviceptr d_dst = reinterpret_cast<gpuDeviceptr>(
        reinterpret_cast<long long unsigned int>(dst) + dst_offset);
    gpuDeviceptr d_src = reinterpret_cast<gpuDeviceptr>(
        reinterpret_cast<long long unsigned int>(src) + src_offset);
    GLOG(gpuMemcpyDtoDAsync(d_dst, d_src, bytes, pimpl_->main_stream_->get()));
}

void GpuManager::memset_d32_async(void *dst, unsigned int val,
                                  size_t num) const {
    gpuDeviceptr d_dst = reinterpret_cast<gpuDeviceptr>(dst);
    GLOG(gpuMemsetD32Async(d_dst, val, num, pimpl_->main_stream_->get()));
}

void GpuManager::memset_d8_async(void *dst, unsigned char val,
                                 size_t num) const {
    gpuDeviceptr d_dst = reinterpret_cast<gpuDeviceptr>(dst);
    GLOG(gpuMemsetD8Async(d_dst, val, num, pimpl_->main_stream_->get()));
}

void GpuManager::sync() const {
    GLOG(gpuStreamSynchronize(pimpl_->main_stream_->get()));
}

}  // namespace ark
