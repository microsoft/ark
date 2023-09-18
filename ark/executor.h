// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef ARK_EXECUTOR_H
#define ARK_EXECUTOR_H

#include "gpu/gpu_kernel.h"
#include "include/ark.h"
#include <memory>

namespace ark {

class Executor::Impl
{
  public:
    Impl(int rank, int world_size, Model &model, const std::string &name,
         int num_warps_per_sm);
    ~Impl();

    void compile();
    void launch();
    void run(int iter);
    void wait();
    float stop();

  private:
    const int rank_;
    const int world_size_;
    int gpu_id_;

    GpuMgrCtx *ctx_;
    std::unique_ptr<BaseScheduler> sched_;
    std::unique_ptr<GpuLoopKernel> glk_;
    GpuStream stream_;
};

} // namespace ark

#endif // ARK_EXECUTOR_H
