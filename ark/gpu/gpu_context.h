#ifndef ARK_GPU_CONTEXT_H_
#define ARK_GPU_CONTEXT_H_

#include <memory>

#include "gpu/gpu_buffer.h"
#include "gpu/gpu_comm_sw.h"
#include "gpu/gpu_manager.h"

namespace ark {
class GpuContext {
   public:
    static std::shared_ptr<GpuContext> get_context(int rank, int world_size);
    std::shared_ptr<GpuManager> get_gpu_manager();
    std::shared_ptr<GpuCommSw> get_comm_sw();
    ~GpuContext() = default;

    std::shared_ptr<GpuBuffer> allocate_buffer(size_t bytes, int align = 1);
    void free_buffer(std::shared_ptr<GpuBuffer> buffer);
    void export_buffer(std::shared_ptr<GpuBuffer> buffer, size_t offset,
                       int expose_id);
    std::shared_ptr<GpuBuffer> import_buffer(size_t bytes, int gpu_id,
                                             int expose_id);
    void freeze(bool expose = false);
    int rank() const;
    int world_size() const;
    int gpu_id() const;
    size_t get_total_bytes() const;
    std::shared_ptr<GpuMemory> get_data_memory(int gpu = -1);

   private:
    GpuContext(int rank, int world_size);
    class Impl;
    std::shared_ptr<Impl> pimpl_;
};

}  // namespace ark
#endif  // ARK_CONTEXT_H_
