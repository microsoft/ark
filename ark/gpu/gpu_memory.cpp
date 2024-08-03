// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "gpu/gpu_memory.h"

#include "gpu/gpu.h"
#include "gpu/gpu_logging.h"
#include "gpu/gpu_manager.h"

namespace ark {

class GpuMemory::Impl {
   public:
    Impl(const GpuManager& manager, size_t bytes, size_t align,
         bool expose = false);
    ~Impl();
    Impl(const Impl&) = delete;
    Impl& operator=(const Impl&) = delete;

   private:
    friend class GpuMemory;

    const GpuManager& manager_;
    size_t bytes_;
    size_t align_;
    size_t bytes_raw_;
    bool is_remote_;
    void* dev_ptr_raw_;
    void* dev_ptr_aligned_;
};

GpuMemory::Impl::Impl(const GpuManager& manager, size_t bytes, size_t align,
                      [[maybe_unused]] bool expose)
    : manager_(manager), bytes_(bytes), align_(align), is_remote_(false) {
    if (bytes_ == 0) {
        dev_ptr_aligned_ = nullptr;
        dev_ptr_raw_ = nullptr;
        return;
    }

    if (align_ == 0) {
        align_ = 1;
    } else if (align_ & (align_ - 1)) {
        ERR(InternalError, "align must be a power of 2. Given %zu.", align_);
    }
    manager_.set_current();
    bytes_raw_ = bytes_ + align_ - 1;
#if defined(ARK_CUDA)
    GLOG(gpuMalloc(&dev_ptr_raw_, bytes_raw_));
#elif defined(ARK_ROCM)
    if (expose) {
        GLOG(hipExtMallocWithFlags(&dev_ptr_raw_, bytes_raw_,
                                   hipDeviceMallocUncached));
    } else {
        GLOG(gpuMalloc(&dev_ptr_raw_, bytes_raw_));
    }
#endif

    // Make sure the raw pointer is a base pointer.
    gpuDeviceptr base_ptr;
    size_t base_size;
    GLOG_DRV(gpuMemGetAddressRange(&base_ptr, &base_size,
                                   (gpuDeviceptr)dev_ptr_raw_));
    if ((void*)base_ptr != dev_ptr_raw_) {
        LOG(ERROR, "unexpected error: dev_ptr_raw_ is not a base pointer.");
    }
    dev_ptr_aligned_ =
        (void*)(((size_t)dev_ptr_raw_ + align_ - 1) & ~(align_ - 1));

    int one = 1;
    GLOG_DRV(gpuPointerSetAttribute(&one, gpuPointerAttributeSyncMemops,
                                    (gpuDeviceptr)dev_ptr_aligned_));

    // Initialize.
    auto stream = manager_.create_stream();
    GLOG(gpuMemsetAsync(dev_ptr_raw_, 0, bytes_raw_, stream->get()));
    stream->sync();
    LOG(DEBUG, "Created GpuMemory addr 0x", std::hex, dev_ptr_aligned_,
        std::dec, " bytes ", bytes_);
}

GpuMemory::Impl::~Impl() {
    if (is_remote_) {
        return;
    }
    if (dev_ptr_raw_ != nullptr) {
        GLOG(gpuFree(dev_ptr_raw_));
    }
}

GpuMemory::GpuMemory(const GpuManager& manager, size_t bytes, size_t align,
                     bool expose)
    : pimpl_(std::make_shared<Impl>(manager, bytes, align, expose)) {}

size_t GpuMemory::bytes() const { return pimpl_->bytes_; }

void* GpuMemory::ref_impl(size_t offset) const {
    return reinterpret_cast<void*>(
        reinterpret_cast<long long unsigned int>(pimpl_->dev_ptr_aligned_) +
        offset);
}

GpuHostMemory::GpuHostMemory(const GpuManager& manager, size_t bytes,
                             unsigned int flags)
    : ptr_(nullptr) {
    manager.set_current();
    GLOG(gpuHostAlloc(&ptr_, bytes, flags));
}

GpuHostMemory::~GpuHostMemory() {
    if (ptr_ != nullptr) {
        GLOG(gpuHostFree(ptr_));
    }
}

}  // namespace ark
