// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "gpu/gpu_manager.h"

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
}

GpuManager::Impl::~Impl() {
    auto e = gpuDevicePrimaryCtxRelease(gpu_dev_);
    if (e != gpuErrorDeinitialized) GLOG(e);
}

std::shared_ptr<GpuManager> GpuManager::get_instance(int gpu_id) {
    return std::shared_ptr<GpuManager>(new GpuManager(gpu_id));
}

GpuManager::GpuManager(int gpu_id)
    : pimpl_(std::make_shared<Impl>(gpu_id)),
      main_stream_(std::make_shared<GpuStreamV2>()) {}

std::shared_ptr<GpuMemory> GpuManager::malloc(size_t bytes, size_t align) {
    return std::make_shared<GpuMemory>(shared_from_this(), bytes, align);
}

std::shared_ptr<GpuStreamV2> GpuManager::main_stream() const {
    return main_stream_;
}

std::shared_ptr<GpuStreamV2> GpuManager::new_stream() {
    this->set_current();
    return std::make_shared<GpuStreamV2>();
}

const GpuManager::Info &GpuManager::info() const { return pimpl_->info_; }

void GpuManager::set_current() const {
    GLOG(gpuCtxSetCurrent(pimpl_->gpu_ctx_));
}

void GpuManager::memcpy_dtoh_async(void *dst, size_t dst_offset, void *src,
                                   size_t src_offset, size_t bytes) const {
    dst = static_cast<char *>(dst) + dst_offset;
    gpuDeviceptr d_src = reinterpret_cast<gpuDeviceptr>(
        reinterpret_cast<uint64_t>(src) + src_offset);
    GLOG(gpuMemcpyDtoHAsync(dst, d_src, bytes, main_stream_->get()));
}

void GpuManager::memcpy_htod_async(void *dst, size_t dst_offset, void *src,
                                   size_t src_offset, size_t bytes) const {
    gpuDeviceptr d_dst = reinterpret_cast<gpuDeviceptr>(
        reinterpret_cast<long long unsigned int>(dst) + dst_offset);
    src = static_cast<char *>(src) + src_offset;
    GLOG(gpuMemcpyHtoDAsync(d_dst, src, bytes, main_stream_->get()));
}

void GpuManager::memcpy_dtod_async(void *dst, size_t dst_offset, void *src,
                                   size_t src_offset, size_t bytes) const {
    gpuDeviceptr d_dst = reinterpret_cast<gpuDeviceptr>(
        reinterpret_cast<long long unsigned int>(dst) + dst_offset);
    gpuDeviceptr d_src = reinterpret_cast<gpuDeviceptr>(
        reinterpret_cast<long long unsigned int>(src) + src_offset);
    GLOG(gpuMemcpyDtoDAsync(d_dst, d_src, bytes, main_stream_->get()));
}

void GpuManager::memset_d32_async(void *dst, unsigned int val,
                                  size_t num) const {
    gpuDeviceptr d_dst = reinterpret_cast<gpuDeviceptr>(dst);
    GLOG(gpuMemsetD32Async(d_dst, val, num, main_stream_->get()));
}

void GpuManager::memset_d8_async(void *dst, unsigned char val,
                                 size_t num) const {
    gpuDeviceptr d_dst = reinterpret_cast<gpuDeviceptr>(dst);
    GLOG(gpuMemsetD8Async(d_dst, val, num, main_stream_->get()));
}

void GpuManager::sync() const {
    GLOG(gpuStreamSynchronize(main_stream_->get()));
}

}  // namespace ark
