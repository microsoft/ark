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
    Impl(const GpuManager& manager,
         const mscclpp::RegisteredMemory& remote_memory);
    ~Impl();
    Impl(const Impl&) = delete;
    Impl& operator=(const Impl&) = delete;

    void to_host(void* dst, size_t offset, size_t bytes, bool async) const;
    void from_host(const void* src, size_t offset, size_t bytes, bool async);
    void from_device(const void* src, size_t offset, size_t bytes, bool async);
    void sync() const;
    void memset(int value, size_t offset, size_t bytes);
    void memset_d32(int value, size_t offset, size_t nelems);

   private:
    friend class GpuMemory;

    const GpuManager& manager_;
    mscclpp::RegisteredMemory remote_memory_;
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
        ERR(InvalidUsageError, "align must be a power of 2. Given %zu.",
            align_);
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
    manager_.memset(dev_ptr_raw_, 0, bytes_raw_);
    LOG(DEBUG, "Created GpuMemory addr 0x", std::hex, dev_ptr_aligned_,
        std::dec, " bytes ", bytes_);
}

GpuMemory::Impl::Impl(const GpuManager& manager,
                      const mscclpp::RegisteredMemory& remote_memory)
    : manager_(manager),
      remote_memory_(remote_memory),
      bytes_(remote_memory_.size()),
      is_remote_(true),
      dev_ptr_raw_(nullptr),
      dev_ptr_aligned_(remote_memory.data()) {}

GpuMemory::Impl::~Impl() {
    if (is_remote_) {
        return;
    }
    if (dev_ptr_raw_ != nullptr) {
        GLOG(gpuFree(dev_ptr_raw_));
    }
}

void GpuMemory::Impl::to_host(void* dst, size_t offset, size_t bytes,
                              bool async) const {
    if (is_remote_) {
        LOG(ERROR, "cannot copy from remote memory.");
    }
    void* dev_ptr = reinterpret_cast<void*>((size_t)dev_ptr_aligned_ + offset);
    manager_.memcpy_dtoh(dst, 0, dev_ptr, 0, bytes, async);
}

void GpuMemory::Impl::from_host(const void* src, size_t offset, size_t bytes,
                                bool async) {
    if (is_remote_) {
        LOG(ERROR, "cannot copy to remote memory.");
    }
    void* dev_ptr = reinterpret_cast<void*>((size_t)dev_ptr_aligned_ + offset);
    manager_.memcpy_htod(dev_ptr, 0, const_cast<void*>(src), 0, bytes, async);
}

void GpuMemory::Impl::from_device(const void* src, size_t offset, size_t bytes,
                                  bool async) {
    if (is_remote_ &&
        !this->remote_memory_.transports().has(mscclpp::Transport::CudaIpc)) {
        LOG(ERROR, "cannot copy to remote memory.");
    }
    void* dev_ptr = reinterpret_cast<void*>((size_t)dev_ptr_aligned_ + offset);
    manager_.memcpy_dtod(dev_ptr, 0, const_cast<void*>(src), 0, bytes, async);
}

void GpuMemory::Impl::sync() const { manager_.sync(); }

void GpuMemory::Impl::memset(int value, size_t offset, size_t bytes) {
    if (is_remote_ &&
        !this->remote_memory_.transports().has(mscclpp::Transport::CudaIpc)) {
        LOG(ERROR, "cannot memset remote memory.");
    }
    manager_.memset(reinterpret_cast<void*>((size_t)dev_ptr_aligned_ + offset),
                    value, bytes);
}

void GpuMemory::Impl::memset_d32(int value, size_t offset, size_t nelems) {
    if (is_remote_ &&
        !this->remote_memory_.transports().has(mscclpp::Transport::CudaIpc)) {
        LOG(ERROR, "cannot memset remote memory.");
    }
    manager_.memset_d32(
        reinterpret_cast<void*>((size_t)dev_ptr_aligned_ + offset), value,
        nelems);
}

GpuMemory::GpuMemory(const GpuManager& manager, size_t bytes, size_t align,
                     bool expose)
    : pimpl_(std::make_shared<Impl>(manager, bytes, align, expose)) {}

GpuMemory::GpuMemory(GpuManager& manager,
                     const mscclpp::RegisteredMemory& remote_memory) {
    pimpl_ = std::make_shared<Impl>(manager, remote_memory);
}

void GpuMemory::resize(size_t bytes, bool expose) {
    size_t align = pimpl_->align_;
    this->pimpl_ =
        std::make_shared<Impl>(pimpl_->manager_, bytes, align, expose);
}

void GpuMemory::resize(const mscclpp::RegisteredMemory& remote_memory) {
    this->pimpl_ = std::make_shared<Impl>(pimpl_->manager_, remote_memory);
}

size_t GpuMemory::bytes() const { return pimpl_->bytes_; }

void GpuMemory::sync() const { pimpl_->sync(); }

void GpuMemory::memset(int value, size_t offset, size_t bytes) {
    pimpl_->memset(value, offset, bytes);
}

void GpuMemory::memset_d32(int value, size_t offset, size_t nelems) {
    pimpl_->memset_d32(value, offset, nelems);
}

void GpuMemory::memcpy_from(const void* src, size_t offset, size_t bytes,
                            bool from_device) {
    if (from_device) {
        pimpl_->from_device(src, offset, bytes, false);
    } else {
        pimpl_->from_host(src, offset, bytes, false);
    }
}

void GpuMemory::memcpy_to(void* dst, size_t offset, size_t bytes) {
    pimpl_->to_host(dst, offset, bytes, false);
}

void* GpuMemory::ref_impl(size_t offset) const {
    return reinterpret_cast<void*>(
        reinterpret_cast<long long unsigned int>(pimpl_->dev_ptr_aligned_) +
        offset);
}

void GpuMemory::to_host(void* dst, size_t bytes, bool async) const {
    pimpl_->to_host(dst, 0, bytes, async);
}

void GpuMemory::from_host(const void* src, size_t bytes, bool async) {
    pimpl_->from_host(src, 0, bytes, async);
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
