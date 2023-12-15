#ifndef ARK_GPU_CONTEXT_H_
#define ARK_GPU_CONTEXT_H_

#include <memory>

#include "gpu/gpu_buffer.h"

namespace ark {

class GpuContext {
   public:
    static std::shared_ptr<GpuContext> get_context(int rank, int world_size);
    ~GpuContext() = default;

    std::shared_ptr<GpuBuffer> allocate_buffer(size_t bytes, int align = 1);
    void free_buffer(std::shared_ptr<GpuBuffer> buffer);
    void export_buffer(std::shared_ptr<GpuBuffer> buffer, size_t offset, int expose_id);
    std::shared_ptr<GpuBuffer> import_buffer(size_t bytes, int gpu_id, int expose_id);
    void freeze(bool expose = false);
    int rank() const;
    int world_size() const;
    void memset(std::shared_ptr<GpuBuffer> buffer, size_t offset, int value,
                size_t bytes);
    void memcpy(std::shared_ptr<GpuBuffer> dst, std::shared_ptr<GpuBuffer> src,
                size_t bytes);

   private:
    GpuContext(int rank, int world_size);
    class Impl;
    std::shared_ptr<Impl> pimpl_;
};

}  // namespace ark
#endif  // ARK_CONTEXT_H_
