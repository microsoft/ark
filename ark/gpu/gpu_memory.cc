// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "gpu/gpu_memory.h"

#include "gpu/gpu.h"
#include "gpu/gpu_logging.h"
#include "gpu/gpu_manager.h"

namespace ark {

class GpuMemory::Impl {
   public:
    Impl(std::shared_ptr<GpuManager> manager, size_t bytes, size_t align);
    ~Impl();
    Impl(const Impl&) = delete;
    Impl& operator=(const Impl&) = delete;

    void to_host(void* dst, bool async) const;
    void from_host(const void* src, size_t bytes, bool async);
    void sync() const;

   private:
    friend class GpuMemory;

    std::shared_ptr<GpuManager> manager_;
    size_t bytes_;
    size_t align_;
    size_t bytes_raw_;
    gpuDeviceptr dev_ptr_raw_;
    gpuDeviceptr dev_ptr_aligned_;
};

GpuMemory::Impl::Impl(std::shared_ptr<GpuManager> manager, size_t bytes,
                      size_t align)
    : manager_(manager), bytes_(bytes), align_(align) {
    if (align_ == 0) {
        align_ = 1;
    } else if (align_ & (align_ - 1)) {
        LOG(ERROR, "align must be a power of 2. Given %zu.", align_);
    }
    manager_->set_current();
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
    manager_->memset_d32_async(reinterpret_cast<void*>(dev_ptr_raw_), 0,
                               bytes_raw_ >> 2);
    manager_->memset_d8_async(reinterpret_cast<void*>((size_t)dev_ptr_raw_ +
                                                      ((bytes_raw_ >> 2) << 2)),
                              0, bytes_raw_ & 3);
    manager_->sync();
}

GpuMemory::Impl::~Impl() { GLOG(gpuMemFree(dev_ptr_raw_)); }

void GpuMemory::Impl::to_host(void* dst, bool async) const {
    void* dev_ptr = reinterpret_cast<void*>(dev_ptr_aligned_);
    manager_->set_current();
    manager_->memcpy_dtoh_async(dst, 0, dev_ptr, 0, bytes_);
    if (!async) {
        manager_->sync();
    }
}

void GpuMemory::Impl::from_host(const void* src, size_t bytes, bool async) {
    void* dev_ptr = reinterpret_cast<void*>(dev_ptr_aligned_);
    manager_->set_current();
    manager_->memcpy_htod_async(dev_ptr, 0, const_cast<void*>(src), 0,
                                    bytes);
    if (!async) {
        manager_->sync();
    }
}

void GpuMemory::Impl::sync() const { manager_->sync(); }

GpuMemory::GpuMemory(std::shared_ptr<GpuManager> manager, size_t bytes,
                     size_t align)
    : pimpl_(std::make_shared<Impl>(manager, bytes, align)) {}

size_t GpuMemory::bytes() const { return pimpl_->bytes_; }

void GpuMemory::sync() const { pimpl_->sync(); }

void GpuMemory::to_host(void* dst, bool async) const {
    pimpl_->to_host(dst, async);
}

void GpuMemory::from_host(const void* src, size_t bytes, bool async) {
    pimpl_->from_host(src, bytes, async);
}

}  // namespace ark
