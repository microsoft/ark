// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef ARK_GPU_MEM_H_
#define ARK_GPU_MEM_H_

#include <cuda.h>
#include <memory>

namespace ark {

typedef CUdeviceptr GpuPtr;

class GpuMem
{
  public:
    struct Info
    {
        // IPC handle.
        CUipcMemHandle ipc_hdl;
        // Physical address of GPU pointer.
        uint64_t phys_addr;
        // Data size.
        uint64_t bytes;
    };

    GpuMem() = default;
    GpuMem(size_t bytes);
    GpuMem(const GpuMem::Info &info);
    ~GpuMem();

    // Allocate a GPU memory chunk.
    void init(size_t bytes);

    // Access the remote GPU memory chunk.
    void init(const GpuMem::Info &info);

    // GPU-side virtual address.
    GpuPtr ref(size_t offset = 0) const;

    // GPU-side physical address.
    uint64_t pref(size_t offset = 0) const;

    // Host-side mapped address.
    void *href(size_t offset = 0) const;

    // Return allocated number of bytes.
    uint64_t get_bytes() const;

    // Return the information for exporting.
    const Info &get_info() const;

  private:
    // Aligned address.
    GpuPtr addr_ = 0;
    // The base address.
    GpuPtr raw_addr_ = 0;
    // Information for exporting.
    Info info_;
    // Number of mmapped 64KB pages.
    uint64_t npage_ = 0;
    // Base address of mmaped pages.
    void *mmap_ = nullptr;
    // True if this object is constructed from a `GpuMem::Info`.
    bool is_remote_;
};

} // namespace ark

#endif // ARK_GPU_MEM_H_
