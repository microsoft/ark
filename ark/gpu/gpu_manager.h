// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef ARK_GPU_MANAGER_H_
#define ARK_GPU_MANAGER_H_

#include <memory>

#include "gpu/gpu.h"
#include "gpu/gpu_common.h"
#include "gpu/gpu_event.h"
#include "gpu/gpu_memory.h"
#include "gpu/gpu_stream.h"

namespace ark {
class GpuManager : public std::enable_shared_from_this<GpuManager> {
   public:
    static std::shared_ptr<GpuManager> get_instance(int gpu_id);
    ~GpuManager() = default;
    GpuManager(const GpuManager &) = delete;
    GpuManager &operator=(const GpuManager &) = delete;

    void set_current() const;
    std::shared_ptr<GpuMemory> malloc(size_t bytes, size_t align = 1,
                                      bool expose = false);
    std::shared_ptr<GpuHostMemory> malloc_host(size_t bytes,
                                               unsigned int flags = 0);
    std::shared_ptr<GpuEventV2> create_event(bool disable_timing = false);

    void memset_d32_sync(void *dst, unsigned int val, size_t num) const;
    void memcpy_htod_sync(void *dst, size_t dst_offset, void *src,
                          size_t src_offset, size_t bytes) const;
    void memcpy_dtoh_sync(void *dst, size_t dst_offset, void *src,
                          size_t src_offset, size_t bytes) const;
    void memcpy_dtod_sync(void *dst, size_t dst_offset, void *src,
                          size_t src_offset, size_t bytes) const;

    int get_gpu_id() const;
    GpuState launch(gpuFunction function, const std::array<int, 3> &grid_dim,
                    const std::array<int, 3> &block_dim, int smem_bytes,
                    std::shared_ptr<GpuStreamV2> stream, void **params,
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
        std::string arch;
    };

   private:
    GpuManager(int gpu_id);
    friend class GpuMemory;

    class Impl;
    std::shared_ptr<Impl> pimpl_;

    void memcpy_dtoh_async(void *dst, size_t dst_offset, void *src,
                           size_t src_offset, size_t bytes) const;
    void memcpy_htod_async(void *dst, size_t dst_offset, void *src,
                           size_t src_offset, size_t bytes) const;
    void memcpy_dtod_async(void *dst, size_t dst_offset, void *src,
                           size_t src_offset, size_t bytes) const;
    void memset_d32_async(void *dst, unsigned int val, size_t num) const;
    void memset_d8_async(void *dst, unsigned char val, size_t num) const;
    void sync() const;
};

}  // namespace ark

#endif  // ARK_GPU_MANAGER_H_
