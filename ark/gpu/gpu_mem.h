// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef ARK_GPU_MEM_H_
#define ARK_GPU_MEM_H_

#include <memory>

#include "gpu/gpu.h"

namespace ark {

typedef gpuDeviceptr GpuPtr;

class GpuMem {
   public:
    struct Info {
        // IPC handle.
        gpuIpcMemHandle ipc_hdl;
        // Data size.
        uint64_t bytes = 0;
    };

    GpuMem() = default;
    GpuMem(size_t bytes);
    GpuMem(const GpuMem::Info &info);
    ~GpuMem();

    // Allocate a GPU memory chunk.
    void init(size_t bytes, bool expose = true);

    // Access the remote GPU memory chunk.
    void init(const GpuMem::Info &info);

    // GPU-side virtual address.
    GpuPtr ref(size_t offset = 0) const;

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
    // True if this object is constructed from a `GpuMem::Info`.
    bool is_remote_;
};

}  // namespace ark

#endif  // ARK_GPU_MEM_H_
