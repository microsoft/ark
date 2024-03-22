// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "ark/executor.h"

#include <algorithm>
#include <string>
#include <memory>

#include "gpu/gpu_loop_kernel.h"
#include "env.h"
#include "logging.h"
#include "sched/sched.h"

namespace ark {

class Executor::Impl {
   public:
    Impl(int rank, int world_size, Model &model, const std::string &name,
         int num_warps_per_sm);
    ~Impl() = default;

    void compile();
    void launch();
    void run(int iter);
    void wait();
    float stop();

   private:
    const int rank_;
    const int world_size_;
    int gpu_id_;

    std::shared_ptr<GpuContext> ctx_;
    std::unique_ptr<BaseScheduler> sched_;
    std::unique_ptr<GpuLoopKernel> glk_;
    std::shared_ptr<GpuStream> stream_;
};

Executor::Impl::Impl(int rank, int world_size, Model &model,
                     const std::string &name, int num_warps_per_sm)
    : rank_{rank}, world_size_{world_size} {
    gpu_id_ = rank_ % get_env().num_ranks_per_host;
    sched_.reset(static_cast<BaseScheduler *>(new DefaultScheduler{
        model, gpu_id_, rank_, world_size_, num_warps_per_sm}));

    sched_->schedule();
    ctx_ = sched_->create_context();
    const GpuManager::Info &ginfo = ctx_->get_gpu_manager()->info();
    stream_ = ctx_->get_gpu_manager()->create_stream();
    glk_ = std::make_unique<GpuLoopKernel>(
        ctx_, name, sched_->gen_code(), ginfo.num_sm, num_warps_per_sm,
        (unsigned int)ginfo.smem_block_total);
}

void Executor::Impl::compile() { glk_->compile(); }

void Executor::Impl::launch() {
    glk_->load();
    glk_->launch(stream_, false);
}

void Executor::Impl::run(int iter) { glk_->run(iter); }

void Executor::Impl::wait() { glk_->wait(); }

float Executor::Impl::stop() {
    glk_->stop();
    return glk_->get_elapsed_msec();
}

Executor::Executor(int rank, int world_size, Model &model,
                   const std::string &name, int num_warps_per_sm)
    : impl_{std::make_unique<Executor::Impl>(rank, world_size, model, name,
                                             num_warps_per_sm)} {}

Executor::~Executor() = default;

void Executor::compile() { impl_->compile(); }

void Executor::launch() { impl_->launch(); }

void Executor::run(int iter) { impl_->run(iter); }

void Executor::wait() { impl_->wait(); }

float Executor::stop() { return impl_->stop(); }

}  // namespace ark
