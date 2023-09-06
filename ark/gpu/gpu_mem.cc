// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include <cassert>
#include <fcntl.h>
#include <fstream>
#include <stdint.h>
#include <string.h>
#include <string>
#include <sys/mman.h>
#include <unistd.h>

#include "gpumemioctl.h"
#define GPUMEM_DRIVER_PATH "/dev/" GPUMEM_DRIVER_NAME

#include "gpu/gpu_logging.h"
#include "gpu/gpu_mem.h"

namespace ark {

bool is_gpumem_loaded()
{
    std::ifstream file("/proc/modules");
    std::string line;
    while (std::getline(file, line)) {
        if (line.find(GPUMEM_DRIVER_NAME) != std::string::npos) {
            return true;
        }
    }
    return false;
}

struct ExposalInfo
{
    // Physical address of GPU pointer.
    uint64_t phys;
    // Number of mmapped 64KB pages.
    uint64_t npage;
    // Base address of mmaped pages.
    void *mmap;
};

// Expose GPU memory space into CPU memory.
static int mem_expose(ExposalInfo *info, GpuPtr addr, uint64_t bytes)
{
    if (!is_gpumem_loaded()) {
        LOG(ERROR, "gpumem driver is not loaded");
    }

    int flag = 1;
    CULOG(cuPointerSetAttribute(&flag, CU_POINTER_ATTRIBUTE_SYNC_MEMOPS, addr));
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
    uint64_t npage = bytes >> 16;
    assert(npage == lock.page_count);
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
    info->mmap =
        mmap(0, bytes, PROT_READ | PROT_WRITE, MAP_SHARED, fd, info->phys);
    if (info->mmap == MAP_FAILED) {
        return errno;
    }
#ifdef DEBUG_ARK_GPU_MEM
    // Test mapping.
    int *tmp0 = (int *)info->mmap;
    *tmp0 = 77;
    int tmp1;
    CULOG(cuMemcpyDtoH(&tmp1, addr, 4));
    if (tmp1 != 77) {
        LOG(ERROR, "mmap test failed: GPU reads ", tmp1, ", expected 77");
    }
    CULOG(cuMemsetD32(addr, 55, 1));
    if (*tmp0 != 55) {
        LOG(ERROR, "mmap test failed: CPU reads ", *tmp0, ", expected 55");
    }
    // Reset the tested address.
    *tmp0 = 0;
#endif // DEBUG_ARK_GPU_MEM
    close(fd);
    return 0;
}

//
static void *map_pa_to_va(uint64_t pa, uint64_t bytes)
{
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

//
GpuMem::GpuMem(size_t bytes)
{
    this->init(bytes);
}

//
GpuMem::GpuMem(const GpuMem::Info &info)
{
    this->init(info);
}

//
void GpuMem::init(size_t bytes)
{
    if (bytes == 0) {
        LOG(ERROR, "Tried to allocate zero byte.");
    }

    // Allocate more to align the bytes by 64KB.
    CULOG(cuMemAlloc(&raw_addr_, bytes + 65536));

    // Make sure it is a base pointer.
    GpuPtr base_ptr;
    size_t base_size; // unused
    CULOG(cuMemGetAddressRange(&base_ptr, &base_size, raw_addr_));
    if (raw_addr_ != base_ptr) {
        LOG(ERROR, "Unexpected error.");
    }

    // Aligned address.
    addr_ = (CUdeviceptr)(((uint64_t)raw_addr_ + 65535) & ~65535);

    ExposalInfo exp_info;
    int err = mem_expose(&exp_info, addr_, bytes);
    if (err != 0) {
        LOG(ERROR, "mem_expose() failed with errno ", err);
    }
    npage_ = exp_info.npage;
    mmap_ = exp_info.mmap;

    CULOG(cuIpcGetMemHandle(&info_.ipc_hdl, raw_addr_));
    info_.phys_addr = exp_info.phys;
    info_.bytes = bytes;

    is_remote_ = false;

    LOG(DEBUG, "Created GpuMem addr 0x", std::hex, addr_, " map ", mmap_,
        std::dec, " bytes ", bytes);
}

//
void GpuMem::init(const GpuMem::Info &info)
{
    // Copy info
    info_.ipc_hdl = info.ipc_hdl;
    info_.phys_addr = info.phys_addr;
    info_.bytes = info.bytes;

    CUresult res = cuIpcOpenMemHandle(&raw_addr_, info.ipc_hdl,
                                      CU_IPC_MEM_LAZY_ENABLE_PEER_ACCESS);
    if (res == CUDA_ERROR_PEER_ACCESS_UNSUPPORTED) {
        LOG(ERROR, "The GPU does not support peer access.");
    } else if (res != CUDA_SUCCESS) {
        // Unexpected error.
        CULOG(res);
    }

    // Aligned address.
    addr_ = (CUdeviceptr)(((uint64_t)raw_addr_ + 65535) & ~65535);

    mmap_ = map_pa_to_va(info.phys_addr, info.bytes);
    if (mmap_ == nullptr) {
        LOG(ERROR, "map_pa_to_va failed");
    }

    // TODO: set npage_

    is_remote_ = true;

    LOG(DEBUG, "Imported GpuMem addr ", std::hex, addr_, " map ", mmap_,
        std::dec, " bytes ", info.bytes);
}

// Destructor.
GpuMem::~GpuMem()
{
    if (is_remote_) {
        CULOG(cuIpcCloseMemHandle(raw_addr_));
    } else {
        CULOG(cuMemFree(raw_addr_));
        if (munmap(mmap_, npage_ << 16) != 0) {
            LOG(ERROR, "munmap failed with errno ", errno);
        }
    }
}

// GPU-side virtual address.
GpuPtr GpuMem::ref(size_t offset) const
{
    return addr_ + offset;
}

// GPU-side physical address.
uint64_t GpuMem::pref(size_t offset) const
{
    return info_.phys_addr + offset;
}

// Host-side mapped address.
void *GpuMem::href(size_t offset) const
{
    return (void *)((char *)mmap_ + offset);
}

// Return allocated number of bytes.
uint64_t GpuMem::get_bytes() const
{
    return info_.bytes;
}

const GpuMem::Info &GpuMem::get_info() const
{
    return info_;
}

} // namespace ark
