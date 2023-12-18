// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "gpu/gpu_memory.h"

#include "gpu/gpu.h"
#include "gpu/gpu_logging.h"
#include "gpu/gpu_manager.h"

namespace ark {

class GpuMemory::Impl {
   public:
    Impl(std::shared_ptr<GpuManager> manager, size_t bytes, size_t align,
         bool expose = false);
    Impl(const mscclpp::RegisteredMemory& remote_memory);
    ~Impl();
    Impl(const Impl&) = delete;
    Impl& operator=(const Impl&) = delete;

    void to_host(void* dst, bool async) const;
    void from_host(const void* src, size_t bytes, bool async);
    void sync() const;

   private:
    friend class GpuMemory;

    std::shared_ptr<GpuManager> manager_;
    mscclpp::RegisteredMemory remote_memory_;
    size_t bytes_;
    size_t align_;
    size_t bytes_raw_;
    bool is_remote_;
    gpuDeviceptr dev_ptr_raw_;
    gpuDeviceptr dev_ptr_aligned_;
};

GpuMemory::Impl::Impl(std::shared_ptr<GpuManager> manager, size_t bytes,
                      size_t align, [[maybe_unused]] bool expose)
    : manager_(manager), bytes_(bytes), align_(align), is_remote_(false) {
    if (bytes_ == 0) {
        dev_ptr_aligned_ = (gpuDeviceptr) nullptr;
        dev_ptr_raw_ = (gpuDeviceptr)(nullptr);
        return;
    }

    if (align_ == 0) {
        align_ = 1;
    } else if (align_ & (align_ - 1)) {
        ERR(InvalidUsageError, "align must be a power of 2. Given %zu.",
            align_);
    }
    manager_->set_current();
    bytes_raw_ = bytes_ + align_ - 1;
#if defined(ARK_CUDA)
    GLOG(gpuMemAlloc(&dev_ptr_raw_, bytes_raw_));
#elif defined(ARK_ROCM)
    if (expose) {
        GLOG(hipExtMallocWithFlags(&dev_ptr_raw_, bytes_raw_,
                                   hipDeviceMallocUncached));
    } else {
        GLOG(gpuMemAlloc(&dev_ptr_raw_, bytes_raw_));
    }
#endif

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
    LOG(DEBUG, "Created GpuMemory addr 0x", std::hex, dev_ptr_aligned_,
        std::dec, " bytes ", bytes_);
}

GpuMemory::Impl::Impl(const mscclpp::RegisteredMemory& remote_memory)
    : remote_memory_(remote_memory),
      bytes_(remote_memory_.size()),
      is_remote_(true),
      dev_ptr_raw_((gpuDeviceptr) nullptr),
      dev_ptr_aligned_((gpuDeviceptr)remote_memory.data()) {}

GpuMemory::Impl::~Impl() {
    if (is_remote_) {
        return;
    }
    if (dev_ptr_raw_ != (gpuDeviceptr) nullptr) {
        GLOG(gpuMemFree(dev_ptr_raw_));
    }
}

void GpuMemory::Impl::to_host(void* dst, bool async) const {
    if (is_remote_) {
        LOG(ERROR, "cannot copy from remote memory.");
    }
    void* dev_ptr = reinterpret_cast<void*>(dev_ptr_aligned_);
    manager_->set_current();
    manager_->memcpy_dtoh_async(dst, 0, dev_ptr, 0, bytes_);
    if (!async) {
        manager_->sync();
    }
}

void GpuMemory::Impl::from_host(const void* src, size_t bytes, bool async) {
    if (is_remote_) {
        LOG(ERROR, "cannot copy to remote memory.");
    }
    void* dev_ptr = reinterpret_cast<void*>(dev_ptr_aligned_);
    manager_->set_current();
    manager_->memcpy_htod_async(dev_ptr, 0, const_cast<void*>(src), 0, bytes);
    if (!async) {
        manager_->sync();
    }
}

void GpuMemory::Impl::sync() const { manager_->sync(); }

GpuMemory::GpuMemory(std::shared_ptr<GpuManager> manager, size_t bytes,
                     size_t align, bool expose)
    : pimpl_(std::make_shared<Impl>(manager, bytes, align, expose)) {}

GpuMemory::GpuMemory(const mscclpp::RegisteredMemory& remote_memory) {
    pimpl_ = std::make_shared<Impl>(remote_memory);
}

void GpuMemory::resize(size_t bytes, bool expose) {
    size_t align = pimpl_->align_;
    this->pimpl_ =
        std::make_shared<Impl>(pimpl_->manager_, bytes, align, expose);
}

void GpuMemory::resize(const mscclpp::RegisteredMemory& remote_memory) {
    this->pimpl_ = std::make_shared<Impl>(remote_memory);
}

GpuPtr GpuMemory::ref(size_t offset) const {
    return reinterpret_cast<GpuPtr>(
        reinterpret_cast<long long unsigned int>(pimpl_->dev_ptr_aligned_) +
        offset);
}

size_t GpuMemory::bytes() const { return pimpl_->bytes_; }

void GpuMemory::sync() const { pimpl_->sync(); }

void GpuMemory::to_host(void* dst, bool async) const {
    pimpl_->to_host(dst, async);
}

void GpuMemory::from_host(const void* src, size_t bytes, bool async) {
    pimpl_->from_host(src, bytes, async);
}

}  // namespace ark
