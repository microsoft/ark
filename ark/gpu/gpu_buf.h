// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef ARK_GPU_BUF_H_
#define ARK_GPU_BUF_H_

#include <memory>

#include "gpu/gpu_mem.h"

namespace ark {

//
class GpuBuf {
   public:
    GpuBuf(int gpu_id, const GpuMem *mem, int id, size_t offset, size_t bytes);

    GpuPtr ref(size_t off = 0) const;

    size_t get_offset() const { return offset; }

    void set_offset(size_t off) { offset = off; }

    int get_gpu_id() const { return gpu_id; }

    const GpuMem *get_mem() const { return mem; }

    int get_id() const { return id; }

    size_t get_bytes() const { return bytes; }

   private:
    int gpu_id;
    const GpuMem *mem;
    // ID of a local buffer or SID of a remote buffer.
    const int id;
    size_t offset;
    size_t bytes;
};

}  // namespace ark

#endif  // ARK_GPU_BUF_H_
