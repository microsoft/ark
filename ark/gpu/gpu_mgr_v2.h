// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef ARK_GPU_MGR_V2_H_
#define ARK_GPU_MGR_V2_H_

#include <memory>

#include "gpu_mem_v2.h"

namespace ark {

class GpuMemV2;
class GpuStreamV2 {
   public:
    GpuStreamV2();
    ~GpuStreamV2() = default;
    void sync() const;

   private:
    friend class GpuMgrV2;
    class Impl;
    std::shared_ptr<Impl> pimpl_;
};

class GpuMgrV2 {
   public:
    GpuMgrV2(int gpu_id);
    ~GpuMgrV2() = default;

    std::shared_ptr<GpuMemV2> malloc(size_t bytes, size_t align = 1);
    std::shared_ptr<GpuStreamV2> main_stream() const;
    std::shared_ptr<GpuStreamV2> new_stream();

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
    friend class GpuMemV2;

    class Impl;
    std::shared_ptr<Impl> pimpl_;
    std::shared_ptr<GpuStreamV2> main_stream_;

    void set_current() const;
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

#endif  // ARK_GPU_MGR_V2_H_
