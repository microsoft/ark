// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "executor.h"

#include <algorithm>
#include <string>

#include "env.h"
#include "include/ark.h"
#include "logging.h"
#include "sched/sched.h"

namespace ark {

Executor::Impl::Impl(int rank, int world_size, Model &model,
                     const std::string &name, int num_warps_per_sm)
    : rank_{rank}, world_size_{world_size} {
    //
    gpu_id_ = rank_ % get_env().num_ranks_per_host;
    sched_.reset(static_cast<BaseScheduler *>(new DefaultScheduler{
        model, gpu_id_, rank_, world_size_, num_warps_per_sm}));

    const GpuInfo &ginfo = get_gpu_mgr(gpu_id_)->get_gpu_info();
    sched_->schedule();
    ctx_ = sched_->create_context(name);
    stream_ = ctx_->create_stream();
    glk_ = std::make_unique<GpuLoopKernel>(
        name, sched_->gen_code(), (unsigned int)ginfo.num_sm,
        (unsigned int)num_warps_per_sm, (unsigned int)ginfo.smem_block_total,
        "", ctx_);
}

Executor::Impl::~Impl() {
    // TODO: pass a shared pointer of GpuMgrCtx to GpuLoopKernel
    // so that we don't need to call reset() here.
    glk_.reset();
    get_gpu_mgr(gpu_id_)->destroy_context(ctx_);
}

void Executor::Impl::compile() {
    glk_->compile(get_gpu_mgr(gpu_id_)->get_gpu_info());
}

void Executor::Impl::launch() {
    glk_->load();
    GpuState ret = glk_->launch(stream_, false);
    if (ret != 0) {
        ERR(ExecutorError, "failed to launch this executor.");
    }
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
