#include <cassert>

#include "ark/gpu/gpu_buf.h"

using namespace std;

namespace ark {

GpuBuf::GpuBuf(const GpuMem *mem_, int id_, size_t offset_, size_t bytes_)
    : mem{mem_}, id{id_}, offset{offset_}, bytes{bytes_}
{
    assert(mem_ != nullptr);
}

GpuPtr GpuBuf::ref(size_t off) const
{
    return this->mem->ref(this->offset + off);
}

uint64_t GpuBuf::pref(size_t off) const
{
    return this->mem->pref(this->offset + off);
}

void *GpuBuf::href(size_t off) const
{
    return this->mem->href(this->offset + off);
}

} // namespace ark
