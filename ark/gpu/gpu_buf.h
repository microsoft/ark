#ifndef ARK_GPU_BUF_H_
#define ARK_GPU_BUF_H_

#include <memory>

#include "ark/gpu/gpu_mem.h"

namespace ark {

//
class GpuBuf
{
  public:
    GpuBuf(const GpuMem *mem, int id, size_t offset, size_t bytes);

    GpuPtr ref(size_t off = 0) const;

    uint64_t pref(size_t off = 0) const;
    void *href(size_t off = 0) const;

    const size_t &get_offset() const
    {
        return offset;
    }
    void set_offset(size_t off)
    {
        offset = off;
    }

    const GpuMem *get_mem() const
    {
        return mem;
    }
    const int &get_id() const
    {
        return id;
    }
    const size_t &get_bytes() const
    {
        return bytes;
    }

  private:
    const GpuMem *mem;
    // ID of a local buffer or SID of a remote buffer.
    int id;
    size_t offset;
    size_t bytes;
};

} // namespace ark

#endif // ARK_GPU_BUF_H_
