// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef ARK_GPU_MEM_H_
#define ARK_GPU_MEM_H_

#include "ipc/ipc_mem.h"
#include <cuda.h>
#include <memory>

namespace ark {

typedef CUdeviceptr GpuPtr;

struct GpuMemInfo
{
    CUipcMemHandle ipc_hdl;
    uint64_t phys_addr;
    uint64_t bytes;
};

// Information on GPU memory exposal to CPU.
struct GpuMemExposalInfo
{
    // Physical address of GPU pointer.
    uint64_t phys = 0;
    // Number of mmapped 64KB pages.
    uint64_t npage = 0;
    // Base address of mmaped pages.
    void *mmap = 0;
};

class GpuMem
{
  public:
    GpuMem(const std::string &name, size_t bytes, bool create,
           bool try_create = false);
    ~GpuMem();

    // Allocate a GPU memory chunk.
    void alloc(size_t bytes);
    // GPU-side virtual address.
    GpuPtr ref(size_t offset = 0) const;
    // GPU-side physical address.
    uint64_t pref(size_t offset = 0) const;
    // Host-side mapped address.
    void *href(size_t offset = 0) const;

    // Return allocated number of bytes.
    const uint64_t &get_bytes() const
    {
        return bytes;
    }

  private:
    //
    std::unique_ptr<IpcMem> shm;
    //
    GpuPtr addr = 0;
    //
    GpuPtr raw_addr = 0;
    //
    uint64_t bytes = 0;
    //
    GpuMemExposalInfo exp_info;
};

} // namespace ark

#endif // ARK_GPU_MEM_H_
