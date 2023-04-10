#include <cassert>
#include <fcntl.h>
#include <stdint.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>

#include "third_party/gpudma/module/gpumemioctl.h"
#define GPUMEM_DRIVER_PATH "/dev/" GPUMEM_DRIVER_NAME

#include "ark/gpu/gpu_logging.h"
#include "ark/gpu/gpu_mem.h"

using namespace std;

namespace ark {

// Expose GPU memory space into CPU memory.
static int mem_expose(GpuMemExposalInfo *info, GpuPtr addr, uint64_t bytes)
{
    int flag = 1;
    CULOG(cuPointerSetAttribute(&flag, CU_POINTER_ATTRIBUTE_SYNC_MEMOPS, addr));
    // Convert virtual into physical address.
    int fd = open(GPUMEM_DRIVER_PATH, O_RDWR, 0);
    if (fd < 0) {
        return errno;
    }
    gpudma_lock_t lock = {.handle = 0, .addr = addr, .size = bytes};
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
        LOGERR("mmap test failed: GPU reads ", tmp1, ", expected 77");
    }
    CULOG(cuMemsetD32(addr, 55, 1));
    if (*tmp0 != 55) {
        LOGERR("mmap test failed: CPU reads ", *tmp0, ", expected 55");
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
        LOGERR("open: ", strerror(errno), " (", errno, ")");
    }
    void *map = mmap(0, bytes, PROT_READ | PROT_WRITE, MAP_SHARED, fd, pa);
    if (map == MAP_FAILED) {
        LOGERR("mmap: ", strerror(errno), " (", errno, ")");
        close(fd);
    }
    close(fd);
    return map;
}

//
GpuMem::GpuMem(const string &name, size_t bytes, bool create, bool try_create)
    : shm{name, create, try_create}
{
    if (bytes > 0) {
        this->alloc(bytes);
    }
}

// Destructor.
GpuMem::~GpuMem()
{
    if (this->shm.is_create()) {
        if (this->addr != 0) {
            cuMemFree(this->addr);
            this->addr = 0;
        }
        if (this->exp_info.mmap != 0) {
            munmap(this->exp_info.mmap, this->exp_info.npage << 16);
            this->exp_info.mmap = 0;
            this->exp_info.npage = 0;
            this->exp_info.phys = 0;
        }
    } else {
        if (this->addr != 0) {
            cuIpcCloseMemHandle(this->addr);
        }
    }
}

//
void GpuMem::alloc(size_t bytes)
{
    // Align the bytes by 64KB.
    this->bytes = ((bytes + 65535) >> 16) << 16;
    if (this->shm.is_create()) {
        if (this->bytes == 0) {
            LOGERR("Tried to allocate zero byte.");
        }
        CULOG(cuMemAlloc(&this->addr, this->bytes));
        int state = mem_expose(&this->exp_info, this->addr, this->bytes);
        if (state != 0) {
            LOGERR("mem_expose() failed with errno ", state);
        }
        //
        IpcLockGuard lg{this->shm.get_lock()};
        GpuMemInfo *info = (GpuMemInfo *)this->shm.alloc(sizeof(GpuMemInfo));
        CULOG(cuIpcGetMemHandle(&info->ipc_hdl, this->addr));
        info->phys_addr = this->exp_info.phys;
        LOG(DEBUG, "Created GpuMem addr ", hex, this->addr, " map ",
            this->exp_info.mmap, dec, " bytes ", this->bytes);
    } else {
        GpuMemInfo *info = (GpuMemInfo *)this->shm.alloc(sizeof(GpuMemInfo));
        IpcLockGuard lg{this->shm.get_lock()};
        CUresult res = cuIpcOpenMemHandle(&this->addr, info->ipc_hdl,
                                          CU_IPC_MEM_LAZY_ENABLE_PEER_ACCESS);
        if (res == CUDA_ERROR_PEER_ACCESS_UNSUPPORTED) {
            // this->addr = 0;
            LOGERR("not implemented yet.");
        } else if (res != CUDA_SUCCESS) {
            // Unexpected error.
            CULOG(res);
        }
        this->exp_info.mmap = map_pa_to_va(info->phys_addr, this->bytes);
        if (this->exp_info.mmap == nullptr) {
            LOGERR("map_pa_to_va failed");
        }
        LOG(DEBUG, "Imported GpuMem addr ", hex, this->addr, " map ",
            this->exp_info.mmap, dec, " bytes ", this->bytes);
    }
}

// GPU-side virtual address.
GpuPtr GpuMem::ref(size_t offset) const
{
    if (this->addr == 0) {
        return 0;
    }
    return this->addr + offset;
}

// GPU-side physical address.
uint64_t GpuMem::pref(size_t offset) const
{
    if (this->exp_info.phys == 0) {
        return 0;
    }
    return this->exp_info.phys + offset;
}

// Host-side mapped address.
void *GpuMem::href(size_t offset) const
{
    if (this->exp_info.mmap == 0) {
        return nullptr;
    }
    return (void *)((char *)this->exp_info.mmap + offset);
}

} // namespace ark
