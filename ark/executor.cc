// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "env.h"
#include "gpu/gpu_kernel.h"
#include "include/ark.h"
#include "include/ark_utils.h"

#include "logging.h"
#include "sched/sched.h"
#include <algorithm>
#include <string>

using namespace std;

namespace ark {

class Executor::Impl
{
  public:
    GpuMgrCtx *ctx;
    BaseScheduler *sched;
    GpuLoopKernel *glk = nullptr;
    GpuStream stream = nullptr;
};

// Constructor.
Executor::Executor(const int gpu_id_, int rank_, int world_size_, Model &model,
                   const string &name, int num_warps_per_sm_)
    : gpu_id{gpu_id_}, rank{rank_},
      world_size{world_size_}, impl{make_unique<Impl>()}
{
    //
    GpuMgr *mgr = get_gpu_mgr(gpu_id);
    const GpuInfo &ginfo = mgr->get_gpu_info();
    if (get_env().scheduler == "Simple") {
        this->impl->sched = new SimpleScheduler{model, gpu_id_, rank_,
                                                world_size_, num_warps_per_sm_};
    }
    if (get_env().scheduler == "Default") {
        this->impl->sched = new DefaultScheduler{
            model, gpu_id_, rank_, world_size_, num_warps_per_sm_};
    }
#ifdef USE_KAHYPAR
    if (get_env().scheduler == "Kahypar") {
        this->impl->sched = new KahyparScheduler{
            model, gpu_id_, rank_, world_size_, num_warps_per_sm_};
    }
#endif // USE_KAHYPAR

    this->impl->sched->schedule();
    this->impl->ctx = this->impl->sched->create_context(name);
    this->impl->stream = this->impl->ctx->create_stream();
    auto codes = this->impl->sched->gen_code();

    this->impl->glk = new GpuLoopKernel{name,
                                        codes,
                                        (unsigned int)ginfo.num_sm,
                                        (unsigned int)num_warps_per_sm_,
                                        (unsigned int)ginfo.smem_block_total,
                                        "",
                                        this->impl->ctx};
}

// Destructor.
Executor::~Executor()
{
    if (this->impl->glk != nullptr) {
        delete this->impl->glk;
    }
    if (this->impl->ctx != nullptr) {
        GpuMgr *mgr = get_gpu_mgr(this->gpu_id);
        mgr->destroy_context(this->impl->ctx);
        this->impl->ctx = nullptr;
    }
}

// Compile the model. This must be called before `launch()`.
void Executor::compile()
{
    GpuMgr *mgr = get_gpu_mgr(gpu_id);
    this->impl->glk->compile(mgr->get_gpu_info());
}

// Launch the model (not running yet). This must be called after `compile()`.
void Executor::launch()
{
    this->impl->glk->load();
    GpuState ret = this->impl->glk->launch(this->impl->stream, false);
    if (ret != 0) {
        LOG(ERROR, "failed to launch this executor.");
    }
}

// Run the model for `iter` iterations.
void Executor::run(int iter)
{
    this->impl->glk->run(iter);
}

// Wait for the previous run to finish.
void Executor::wait()
{
    this->impl->glk->wait();
}

// Stop the model and return the elapsed time in milliseconds.
// Once this is called, we need to call `launch()` again to run the model again.
float Executor::stop()
{
    this->impl->glk->stop();
    return this->impl->glk->get_elapsed_msec();
}

} // namespace ark
