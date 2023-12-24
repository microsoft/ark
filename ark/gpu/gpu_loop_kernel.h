// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef ARK_GPU_LOOP_KERNEL_H_
#define ARK_GPU_LOOP_KERNEL_H_

#include <memory>

#include "gpu/gpu_kernel.h"

#define ARK_BUF_NAME "ARK_BUF"
#define ARK_LSS_NAME "ARK_LOOP_SYNC_STATE"

namespace ark {

class GpuLoopKernel : public GpuKernel {
   public:
    GpuLoopKernel(std::shared_ptr<GpuContext> ctx, const std::string &name,
                  const std::vector<std::string> &codes, int num_sm,
                  int num_warp, unsigned int smem_bytes);

    GpuState launch(std::shared_ptr<GpuStream> stream,
                    bool disable_timing = true);
    void load();
    void run(int iter = 1);
    bool poll();
    void wait();
    void stop();

    float get_elapsed_msec() const;

   private:
    std::shared_ptr<GpuEvent> timer_begin_;
    std::shared_ptr<GpuEvent> timer_end_;

    int threads_per_warp_ = -1;
    std::shared_ptr<GpuHostMemory> flag_ = nullptr;

    std::shared_ptr<GpuStream> stream_ = nullptr;
    bool is_recording_ = false;
    float elapsed_msec_ = -1;
};
}  // namespace ark

#endif  // ARK_GPU_LOOP_KERNEL_H_
