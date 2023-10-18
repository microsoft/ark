// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "gpu/gpu_buf.h"

#include <cassert>

using namespace std;

namespace ark {

GpuBuf::GpuBuf(int gpu_id_, const GpuMem *mem_, int id_, size_t offset_,
               size_t bytes_)
    : gpu_id{gpu_id_}, mem{mem_}, id{id_}, offset{offset_}, bytes{bytes_} {
    assert(mem_ != nullptr);
}

GpuPtr GpuBuf::ref(size_t off) const {
    return this->mem->ref(this->offset + off);
}

uint64_t GpuBuf::pref(size_t off) const {
    return this->mem->pref(this->offset + off);
}

void *GpuBuf::href(size_t off) const {
    return this->mem->href(this->offset + off);
}

}  // namespace ark
