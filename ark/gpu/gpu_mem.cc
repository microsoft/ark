// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "gpu/gpu_mem.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

#include <cassert>
#include <cstring>
#include <fstream>
#include <string>

#include "gpu/gpu_logging.h"

#if defined(ARK_ROCM)
#include <rocm-core/rocm_version.h>
#endif  // defined(ARK_ROCM)

#define GPU_PAGE_SHIFT 16
#define GPU_PAGE_SIZE (1ULL << GPU_PAGE_SHIFT)
#define GPU_PAGE_OFFSET (GPU_PAGE_SIZE - 1)
#define GPU_PAGE_MASK (~GPU_PAGE_OFFSET)

namespace ark {

#if defined(ARK_CUDA)

#include "gpumemioctl.h"
#define GPUMEM_DRIVER_PATH "/dev/" GPUMEM_DRIVER_NAME

bool is_gpumem_loaded() {
    std::ifstream file("/proc/modules");
    std::string line;
    while (std::getline(file, line)) {
        if (line.find(GPUMEM_DRIVER_NAME) != std::string::npos) {
            return true;
        }
    }
    return false;
}

struct ExposalInfo {
    // Physical address of GPU pointer.
    uint64_t phys;
    // Number of mmapped 64KB pages.
    uint64_t npage;
    // Base address of mmaped pages.
    void *mmap;
};

// Expose GPU memory space into CPU memory.
static int mem_expose(ExposalInfo *info, GpuPtr addr, uint64_t bytes) {
    if (!is_gpumem_loaded()) {
        LOG(ERROR, "gpumem driver is not loaded");
    }

    // Convert virtual into physical address.
    int fd = open(GPUMEM_DRIVER_PATH, O_RDWR, 0);
    if (fd < 0) {
        return errno;
    }
    gpudma_lock_t lock;
    lock.handle = 0;
    lock.addr = addr;
    lock.size = bytes;
    if (ioctl(fd, IOCTL_GPUMEM_LOCK, &lock) < 0) {
        return errno;
    }
    uint64_t npage = bytes >> GPU_PAGE_SHIFT;
    // +1 can happen as we alloc 64KB more for safe alignment.
    if (npage != lock.page_count && npage + 1 != lock.page_count) {
        LOG(ERROR, "Unexpected number of pages: ", npage, " vs ",
            lock.page_count);
    }
    npage = lock.page_count;
    int state_bytes = sizeof(gpudma_state_t) + npage * sizeof(uint64_t);
    gpudma_state_t *state = (gpudma_state_t *)malloc(state_bytes);
    if (state == 0) {
        return errno;
    }
    memset(state, 0, state_bytes);
    state->handle = lock.handle;
    state->page_count = npage;
    if (ioctl(fd, IOCTL_GPUMEM_STATE, state) < 0) {
        return errno;
    }
    // Set the physical address.
    info->phys = state->pages[0];
    info->npage = npage;
    free(state);
    // Create mmap of all pages.
    info->mmap = mmap(0, npage << GPU_PAGE_SHIFT, PROT_READ | PROT_WRITE,
                      MAP_SHARED, fd, info->phys);
    if (info->mmap == MAP_FAILED) {
        return errno;
    }
#ifdef DEBUG_ARK_GPU_MEM
    // Test mapping.
    int *tmp0 = (int *)info->mmap;
    *tmp0 = 77;
    int tmp1;
    GLOG(gpuMemcpyDtoH(&tmp1, addr, 4));
    if (tmp1 != 77) {
        LOG(ERROR, "mmap test failed: GPU reads ", tmp1, ", expected 77");
    }
    GLOG(gpuMemsetD32(addr, 55, 1));
    if (*tmp0 != 55) {
        LOG(ERROR, "mmap test failed: CPU reads ", *tmp0, ", expected 55");
    }
    // Reset the tested address.
    *tmp0 = 0;
#endif  // DEBUG_ARK_GPU_MEM
    close(fd);
    return 0;
}

//
static void *map_pa_to_va(uint64_t pa, uint64_t bytes) {
    int fd = open(GPUMEM_DRIVER_PATH, O_RDWR, 0);
    if (fd < 0) {
        LOG(ERROR, "open: ", strerror(errno), " (", errno, ")");
    }
    void *map = mmap(0, bytes, PROT_READ | PROT_WRITE, MAP_SHARED, fd, pa);
    if (map == MAP_FAILED) {
        LOG(ERROR, "mmap: ", strerror(errno), " (", errno, ")");
        close(fd);
    }
    close(fd);
    return map;
}

#endif  // defined(ARK_CUDA)

//
GpuMem::GpuMem(size_t bytes) { this->init(bytes); }

//
GpuMem::GpuMem(const GpuMem::Info &info) { this->init(info); }

//
void GpuMem::init(size_t bytes, bool expose) {
    if (bytes == 0) {
        LOG(ERROR, "Tried to allocate zero byte.");
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
        LOG(ERROR, "Unexpected error.");
    }

    // Aligned address.
    addr_ =
        (gpuDeviceptr)(((uint64_t)raw_addr_ + GPU_PAGE_OFFSET) & GPU_PAGE_MASK);

    int one = 1;
    GLOG(gpuPointerSetAttribute(&one, gpuPointerAttributeSyncMemops, addr_));

#if defined(ARK_CUDA)
    ExposalInfo exp_info;
    if (expose) {
        int err = mem_expose(&exp_info, addr_, bytes + GPU_PAGE_SIZE);
        if (err != 0) {
            LOG(ERROR, "mem_expose() failed with errno ", err);
        }
    } else {
        exp_info.npage = 0;
        exp_info.mmap = nullptr;
        exp_info.phys = 0;
    }
    npage_ = exp_info.npage;
    mmap_ = exp_info.mmap;
    info_.phys_addr = exp_info.phys;
#elif defined(ARK_ROCM)
    // Alignment diff.
    uint64_t diff = (uint64_t)addr_ - (uint64_t)raw_addr_;
    npage_ = (bytes - diff + GPU_PAGE_SIZE) >> GPU_PAGE_SHIFT;
    mmap_ = addr_;
    // Just set to virtual address.
    info_.phys_addr = reinterpret_cast<uint64_t>(addr_);
#endif

#if defined(ARK_CUDA) || (defined(ARK_ROCM) && (ROCM_VERSION_MAJOR == 5 &&  \
                                                ROCM_VERSION_MINOR >= 7) || \
                          (ROCM_VERSION_MAJOR > 5))
    GLOG(gpuIpcGetMemHandle(&info_.ipc_hdl, raw_addr_));
#endif
    info_.bytes = bytes;

    is_remote_ = false;

    LOG(DEBUG, "Created GpuMem addr 0x", std::hex, addr_, " map ", mmap_,
        std::dec, " bytes ", bytes);
}

//
void GpuMem::init(const GpuMem::Info &info) {
    // Copy info
    info_.ipc_hdl = info.ipc_hdl;
    info_.phys_addr = info.phys_addr;
    info_.bytes = info.bytes;

    gpuError res = gpuIpcOpenMemHandle(&raw_addr_, info.ipc_hdl,
                                       gpuIpcMemLazyEnablePeerAccess);
    if (res == gpuErrorPeerAccessUnsupported) {
        LOG(ERROR, "The GPU does not support peer access.");
    } else if (res != gpuSuccess) {
        // Unexpected error.
        GLOG(res);
    }

    // Aligned address.
    addr_ =
        (gpuDeviceptr)(((uint64_t)raw_addr_ + GPU_PAGE_OFFSET) & GPU_PAGE_MASK);

    if (info.phys_addr != 0) {
#if defined(ARK_CUDA)
        mmap_ = map_pa_to_va(info.phys_addr, info.bytes);
        if (mmap_ == nullptr) {
            LOG(ERROR, "map_pa_to_va failed");
        }
#elif defined(ARK_ROCM)
        mmap_ = reinterpret_cast<void *>(info.phys_addr);
#endif
    } else {
        mmap_ = nullptr;
    }

    // TODO: set npage_

    is_remote_ = true;

    LOG(DEBUG, "Imported GpuMem addr 0x", std::hex, addr_, " map ", mmap_,
        std::dec, " bytes ", info.bytes);
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
#if defined(ARK_CUDA)
    if (mmap_ != nullptr) {
        size_t mapped_bytes;
        if (is_remote_) {
            mapped_bytes = info_.bytes;
        } else {
            mapped_bytes = npage_ << GPU_PAGE_SHIFT;
        }
        munmap(mmap_, mapped_bytes);
    }
#endif
}

// GPU-side virtual address.
GpuPtr GpuMem::ref(size_t offset) const {
    return reinterpret_cast<GpuPtr>(
        reinterpret_cast<long long unsigned int>(addr_) + offset);
}

// GPU-side physical address.
uint64_t GpuMem::pref(size_t offset) const { return info_.phys_addr + offset; }

// Host-side mapped address.
void *GpuMem::href(size_t offset) const {
    return (void *)((char *)mmap_ + offset);
}

// Return allocated number of bytes.
uint64_t GpuMem::get_bytes() const { return info_.bytes; }

const GpuMem::Info &GpuMem::get_info() const { return info_; }

}  // namespace ark
