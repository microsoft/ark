// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "gpu/gpu_mgr_v2.h"

#include "gpu/gpu.h"
#include "gpu/gpu_logging.h"

namespace ark {

class GpuMgrV2::Impl {
   public:
    Impl(int gpu_id);
    ~Impl();

   private:
    friend class GpuMgrV2;

    int gpu_id_;
    gpuDevice gpu_dev_;
    gpuCtx gpu_ctx_;
    gpuStream gpu_stream_;
    GpuMgrV2::Info info_;
};

GpuMgrV2::Impl::Impl(int gpu_id) : gpu_id_(gpu_id) {
    if (gpuDeviceGet(&gpu_dev_, gpu_id) == gpuErrorNotInitialized) {
        GLOG(gpuInit(0));
        GLOG(gpuDeviceGet(&gpu_dev_, gpu_id));
    }
    GLOG(gpuDevicePrimaryCtxRetain(&gpu_ctx_, gpu_dev_));
    GLOG(gpuCtxSetCurrent(gpu_ctx_));

    GLOG(gpuStreamCreate(&gpu_stream_, gpuStreamNonBlocking));

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

GpuMgrV2::Impl::~Impl() {
    GLOG(gpuStreamDestroy(gpu_stream_));
    auto e = gpuDevicePrimaryCtxRelease(gpu_dev_);
    if (e != gpuErrorDeinitialized) GLOG(e);
}

////////////////////////////////////////////////////////////////////////////////

GpuMgrV2::GpuMgrV2(int gpu_id) : pimpl_(std::make_shared<Impl>(gpu_id)) {}

GpuMgrV2::~GpuMgrV2() = default;

std::shared_ptr<GpuMemV2> GpuMgrV2::malloc(size_t bytes, size_t align) {
    return std::make_shared<GpuMemV2>(*this, bytes, align);
}

const GpuMgrV2::Info &GpuMgrV2::info() const { return pimpl_->info_; }

void GpuMgrV2::set_current() const { GLOG(gpuCtxSetCurrent(pimpl_->gpu_ctx_)); }

void GpuMgrV2::memcpy_dtoh_async(void *dst, size_t dst_offset, void *src,
                                 size_t src_offset, size_t bytes) const {
    dst = static_cast<char *>(dst) + dst_offset;
    gpuDeviceptr d_src = reinterpret_cast<gpuDeviceptr>(
        reinterpret_cast<long long unsigned int>(src) + src_offset);
    GLOG(gpuMemcpyDtoHAsync(dst, d_src, bytes, pimpl_->gpu_stream_));
}

void GpuMgrV2::memcpy_htod_async(void *dst, size_t dst_offset, const void *src,
                                 size_t src_offset, size_t bytes) const {
    gpuDeviceptr d_dst = reinterpret_cast<gpuDeviceptr>(
        reinterpret_cast<long long unsigned int>(dst) + dst_offset);
    src = static_cast<const char *>(src) + src_offset;
    GLOG(gpuMemcpyHtoDAsync(d_dst, src, bytes, pimpl_->gpu_stream_));
}

void GpuMgrV2::memcpy_dtod_async(void *dst, size_t dst_offset, void *src,
                                 size_t src_offset, size_t bytes) const {
    gpuDeviceptr d_dst = reinterpret_cast<gpuDeviceptr>(
        reinterpret_cast<long long unsigned int>(dst) + dst_offset);
    gpuDeviceptr d_src = reinterpret_cast<gpuDeviceptr>(
        reinterpret_cast<long long unsigned int>(src) + src_offset);
    GLOG(gpuMemcpyDtoDAsync(d_dst, d_src, bytes, pimpl_->gpu_stream_));
}

void GpuMgrV2::memset_d32_async(void *dst, unsigned int val, size_t num) const {
    gpuDeviceptr d_dst = reinterpret_cast<gpuDeviceptr>(dst);
    GLOG(gpuMemsetD32Async(d_dst, val, num, pimpl_->gpu_stream_));
}

void GpuMgrV2::memset_d8_async(void *dst, unsigned char val, size_t num) const {
    gpuDeviceptr d_dst = reinterpret_cast<gpuDeviceptr>(dst);
    GLOG(gpuMemsetD8Async(d_dst, val, num, pimpl_->gpu_stream_));
}

void GpuMgrV2::sync() const { GLOG(gpuStreamSynchronize(pimpl_->gpu_stream_)); }

}  // namespace ark
