// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "gpu/gpu_mem.h"

#include <cassert>
#include <cstring>
#include <fstream>
#include <string>

#include "gpu/gpu_logging.h"
#include "include/ark.h"

#define GPU_PAGE_SHIFT 16
#define GPU_PAGE_SIZE (1ULL << GPU_PAGE_SHIFT)
#define GPU_PAGE_OFFSET (GPU_PAGE_SIZE - 1)
#define GPU_PAGE_MASK (~GPU_PAGE_OFFSET)

namespace ark {

//
GpuMem::GpuMem(size_t bytes) { this->init(bytes); }

//
GpuMem::GpuMem(const GpuMem::Info &info) { this->init(info); }

//
void GpuMem::init(size_t bytes, bool expose) {
    if (bytes == 0) {
        ERR(InvalidUsageError, "Tried to allocate zero byte.");
    }

#if defined(ARK_CUDA)
    // Allocate more to align the bytes by 64KB.
    GLOG(gpuMemAlloc(&raw_addr_, bytes + GPU_PAGE_SIZE));
#elif defined(ARK_ROCM)
    if (expose) {
        GLOG(hipExtMallocWithFlags(&raw_addr_, bytes + GPU_PAGE_SIZE,
                                   hipDeviceMallocUncached));
    } else {
        GLOG(gpuMemAlloc(&raw_addr_, bytes + GPU_PAGE_SIZE));
    }
#endif

    // Make sure it is a base pointer.
    GpuPtr base_ptr;
    size_t base_size;  // unused
    GLOG(gpuMemGetAddressRange(&base_ptr, &base_size, raw_addr_));
    if (raw_addr_ != base_ptr) {
        ERR(ExecutorError, "Unexpected error.");
    }

    // Aligned address.
    addr_ =
        (gpuDeviceptr)(((uint64_t)raw_addr_ + GPU_PAGE_OFFSET) & GPU_PAGE_MASK);

    int one = 1;
    GLOG(gpuPointerSetAttribute(&one, gpuPointerAttributeSyncMemops, addr_));

#if defined(ARK_CUDA) || (defined(ARK_ROCM) && (HIP_VERSION >= 50700000))
    GLOG(gpuIpcGetMemHandle(&info_.ipc_hdl, raw_addr_));
#endif
    info_.bytes = bytes;

    is_remote_ = false;

    LOG(DEBUG, "Created GpuMem addr 0x", std::hex, addr_, std::dec, " bytes ",
        bytes);
}

//
void GpuMem::init(const GpuMem::Info &info) {
    // Copy info
    info_.ipc_hdl = info.ipc_hdl;
    info_.bytes = info.bytes;

    gpuError res = gpuIpcOpenMemHandle(&raw_addr_, info.ipc_hdl,
                                       gpuIpcMemLazyEnablePeerAccess);
    if (res == gpuErrorPeerAccessUnsupported) {
        ERR(SystemError, "The GPU does not support peer access.");
    } else if (res != gpuSuccess) {
        // Unexpected error.
        GLOG(res);
    }

    // Aligned address.
    addr_ =
        (gpuDeviceptr)(((uint64_t)raw_addr_ + GPU_PAGE_OFFSET) & GPU_PAGE_MASK);

    is_remote_ = true;

    LOG(DEBUG, "Imported GpuMem addr 0x", std::hex, addr_, std::dec, " bytes ",
        info.bytes);
}

// Destructor.
GpuMem::~GpuMem() {
    if (addr_ == 0) {
        return;
    }
    if (is_remote_) {
        if (gpuIpcCloseMemHandle(raw_addr_) != gpuSuccess) {
            LOG(WARN, "gpuIpcCloseMemHandle() failed.");
        }
    } else {
        if (gpuMemFree(raw_addr_) != gpuSuccess) {
            LOG(WARN, "gpuMemFree() failed.");
        }
    }
}

// GPU-side virtual address.
GpuPtr GpuMem::ref(size_t offset) const {
    return reinterpret_cast<GpuPtr>(
        reinterpret_cast<long long unsigned int>(addr_) + offset);
}

// Return allocated number of bytes.
uint64_t GpuMem::get_bytes() const { return info_.bytes; }

const GpuMem::Info &GpuMem::get_info() const { return info_; }

}  // namespace ark
