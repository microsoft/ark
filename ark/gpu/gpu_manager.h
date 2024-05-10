// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef ARK_GPU_MANAGER_H_
#define ARK_GPU_MANAGER_H_

#include <memory>

#include "arch.hpp"
#include "gpu/gpu.h"
#include "gpu/gpu_event.h"
#include "gpu/gpu_memory.h"
#include "gpu/gpu_stream.h"

namespace ark {
class GpuManager {
   public:
    static std::shared_ptr<GpuManager> get_instance(int gpu_id);

    GpuManager(const GpuManager &) = delete;
    ~GpuManager() = default;
    GpuManager &operator=(const GpuManager &) = delete;

    void set_current() const;
    std::shared_ptr<GpuMemory> malloc(size_t bytes, size_t align = 1,
                                      bool expose = false);
    std::shared_ptr<GpuHostMemory> malloc_host(size_t bytes,
                                               unsigned int flags = 0);
    std::shared_ptr<GpuEvent> create_event(bool disable_timing = false) const;
    std::shared_ptr<GpuStream> create_stream() const;

    int get_gpu_id() const;
    void launch(gpuFunction function, const std::array<int, 3> &grid_dim,
                const std::array<int, 3> &block_dim, int smem_bytes,
                std::shared_ptr<GpuStream> stream, void **params,
                void **extra) const;

    struct Info;
    const Info &info() const;

    struct Info {
        int cc_major;
        int cc_minor;
        size_t gmem_total;
        int smem_total;
        int smem_block_total;
        int num_sm;
        int clk_rate;
        int threads_per_warp;
        int max_registers_per_block;
        int max_threads_per_block;
        int max_registers_per_thread = 256;  // TODO: how to get this?
        int min_threads_per_block =
            max_registers_per_block / max_registers_per_thread;
        int smem_align = 128;  // TODO: how to get this?
        Arch arch;
    };

   private:
    GpuManager(int gpu_id);

    class Impl;
    std::shared_ptr<Impl> pimpl_;
};

}  // namespace ark

#endif  // ARK_GPU_MANAGER_H_
