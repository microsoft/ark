// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "gpu/gpu_mem_v2.h"

#include "gpu/gpu.h"
#include "gpu/gpu_logging.h"
#include "gpu/gpu_mgr_v2.h"

namespace ark {

class GpuMemV2::Impl {
   public:
    explicit Impl(GpuMgrV2& gpu_mgr, size_t bytes, size_t align);
    ~Impl();

    void to_host(void* dst, bool async) const;
    void from_host(const void* src, size_t bytes, bool async);
    void sync() const;

   private:
    friend class GpuMemV2;

    std::shared_ptr<GpuMgrV2> gpu_mgr_;
    size_t bytes_;
    size_t align_;
    size_t bytes_raw_;
    gpuDeviceptr dev_ptr_raw_;
    gpuDeviceptr dev_ptr_aligned_;
};

GpuMemV2::Impl::Impl(GpuMgrV2& gpu_mgr, size_t bytes, size_t align)
    : gpu_mgr_(&gpu_mgr), bytes_(bytes), align_(align) {
    if (align_ == 0) {
        align_ = 1;
    } else if (align_ & (align_ - 1)) {
        LOG(ERROR, "align must be a power of 2. Given %zu.", align_);
    }
    gpu_mgr_->set_current();
    bytes_raw_ = bytes_ + align_ - 1;
    GLOG(gpuMemAlloc(&dev_ptr_raw_, bytes_raw_));

    // Make sure the raw pointer is a base pointer.
    gpuDeviceptr base_ptr;
    size_t base_size;
    GLOG(gpuMemGetAddressRange(&base_ptr, &base_size, dev_ptr_raw_));
    if (base_ptr != dev_ptr_raw_) {
        LOG(ERROR, "unexpected error: dev_ptr_raw_ is not a base pointer.");
    }
    dev_ptr_aligned_ =
        (gpuDeviceptr)(((size_t)dev_ptr_raw_ + align_ - 1) & ~(align_ - 1));

    int one = 1;
    GLOG(gpuPointerSetAttribute(&one, gpuPointerAttributeSyncMemops,
                                dev_ptr_aligned_));

    // Initialize.
    gpu_mgr_->memset_d32_async(reinterpret_cast<void*>(dev_ptr_raw_), 0,
                               bytes_raw_ >> 2);
    gpu_mgr_->memset_d8_async(reinterpret_cast<void*>((size_t)dev_ptr_raw_ +
                                                      ((bytes_raw_ >> 2) << 2)),
                              0, bytes_raw_ & 3);
    gpu_mgr_->sync();
}

GpuMemV2::Impl::~Impl() { GLOG(gpuMemFree(dev_ptr_raw_)); }

void GpuMemV2::Impl::to_host(void* dst, bool async) const {
    void* dev_ptr = reinterpret_cast<void*>(dev_ptr_aligned_);
    gpu_mgr_->set_current();
    gpu_mgr_->memcpy_dtoh_async(dst, 0, dev_ptr, 0, bytes_);
    if (!async) {
        gpu_mgr_->sync();
    }
}

void GpuMemV2::Impl::from_host(const void* src, size_t bytes, bool async) {
    void* dev_ptr = reinterpret_cast<void*>(dev_ptr_aligned_);
    gpu_mgr_->set_current();
    gpu_mgr_->memcpy_htod_async(dev_ptr, 0, src, 0, bytes);
    if (!async) {
        gpu_mgr_->sync();
    }
}

void GpuMemV2::Impl::sync() const { gpu_mgr_->sync(); }

////////////////////////////////////////////////////////////////////////////////

GpuMemV2::GpuMemV2(GpuMgrV2& gpu_mgr, size_t bytes, size_t align)
    : pimpl_(std::make_shared<Impl>(gpu_mgr, bytes, align)) {}

size_t GpuMemV2::bytes() const { return pimpl_->bytes_; }

void GpuMemV2::sync() const { pimpl_->sync(); }

void GpuMemV2::to_host(void* dst, bool async) const {
    pimpl_->to_host(dst, async);
}

void GpuMemV2::from_host(const void* src, size_t bytes, bool async) {
    pimpl_->from_host(src, bytes, async);
}

}  // namespace ark
