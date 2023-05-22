// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "ark/env.h"
#include "ark/gpu/gpu_kernel.h"
#include "ark/include/ark.h"
#include "ark/logging.h"
#include "ark/sched/sched.h"
#include <algorithm>
#include <string>
using namespace std;

namespace ark {

class ExecutorMember
{
  public:
    GpuMgrCtx *ctx;
    SchedulerBase *sched;
    GpuLoopKernel *glk = nullptr;
    GpuStream stream = nullptr;
};

// Constructor.
Executor::Executor(const int gpu_id_, int rank_, int world_size_,
                   const Model &model, const string &name)
    : gpu_id{gpu_id_}, rank{rank_},
      world_size{world_size_}, member{make_unique<ExecutorMember>()}
{
    //
    GpuMgr *mgr = get_gpu_mgr(gpu_id);
    const GpuInfo &ginfo = mgr->get_gpu_info();
    if (get_env().scheduler == "Simple") {
        this->member->sched =
            new SimpleScheduler{gpu_id_, rank_, world_size_, model};
    }
    if (get_env().scheduler == "Default") {
        this->member->sched =
            new DefaultScheduler{gpu_id_, rank_, world_size_, model};
    }
#ifdef USE_KAHYPAR
    if (get_env().scheduler == "Kahypar") {
        this->member->sched =
            new KahyparScheduler{gpu_id_, rank_, world_size_, model};
    }
#endif // USE_KAHYPAR

    this->member->ctx = this->member->sched->create_context(name);
    this->member->stream = this->member->ctx->create_stream();
    auto codes = this->member->sched->schedule();
    unsigned int num_depths = this->member->sched->get_num_depths();

    this->member->glk = new GpuLoopKernel{name,
                                          codes,
                                          (unsigned int)ginfo.num_sm,
                                          16,
                                          (unsigned int)ginfo.smem_block_total,
                                          "",
                                          this->member->ctx,
                                          num_depths};
}

// Destructor.
Executor::~Executor()
{
    if (this->member->glk != nullptr) {
        delete this->member->glk;
    }
    if (this->member->ctx != nullptr) {
        GpuMgr *mgr = get_gpu_mgr(this->gpu_id);
        mgr->destroy_context(this->member->ctx);
        this->member->ctx = nullptr;
    }
}

// Compile the model. This must be called before `launch()`.
void Executor::compile()
{
    GpuMgr *mgr = get_gpu_mgr(gpu_id);
    this->member->glk->compile(mgr->get_gpu_info());
}

// Launch the model (not running yet). This must be called after `compile()`.
void Executor::launch()
{
    this->member->glk->load();
    GpuState ret = this->member->glk->launch(this->member->stream, false);
    if (ret != 0) {
        LOGERR("failed to launch this executor.");
    }
}

// Run the model for `iter` iterations.
void Executor::run(int iter)
{
    this->member->glk->run(iter);
}

// Wait for the previous run to finish.
void Executor::wait()
{
    this->member->glk->wait();
}

// Stop the model and return the elapsed time in milliseconds.
// Once this is called, we need to call `launch()` again to run the model again.
float Executor::stop()
{
    this->member->glk->stop();
    return this->member->glk->get_elapsed_msec();
}

// Get the corresponding tensor of the executor from the given model tensor.
// Both tensors may be different if the scheduler creates an optimized model
// out of the original one.
Tensor *Executor::get_tensor(Tensor *tns) const
{
    return this->member->sched->get_tensor(tns);
}

// Get the corresponding GPU buffer of the executor from the given model tensor.
GpuBuf *Executor::get_gpu_buf(Tensor *tns) const
{
    return this->member->sched->get_gpu_buf(tns);
}

// Copy contiguous data from a host buffer to the given tensor's (possibly
// non-contiguous) data range on GPU.
void Executor::tensor_memcpy(Tensor *dst, const void *src, size_t bytes)
{
    GpuBuf *buf = this->get_gpu_buf(dst);
    if (buf == nullptr) {
        LOGERR("failed to get GPU buffer for tensor ", dst->id);
    }
    Tensor *tns = this->get_tensor(dst);
    if (bytes > (size_t)tns->shape_bytes()) {
        LOGERR("the given number of bytes (", bytes,
               ") is larger than the tensor size (", tns->shape_bytes(), ")");
    }
    int ndims = tns->ndims();
    char *ps = (char *)src;
    if (ndims == 1) {
        gpu_memcpy(buf->ref(tns->offset_bytes(0)), ps, bytes);
        return;
    }
    size_t done = 0;
    size_t rem = bytes;
    for (DimType i = 0; i < tns->shape[0]; ++i) {
        if (ndims == 2) {
            size_t cb = min(rem, (size_t)tns->shape[1] * tns->type_bytes());
            gpu_memcpy(buf->ref(tns->offset_bytes(i, 0)), &ps[done], cb);
            rem -= cb;
            done += cb;
            if (rem == 0) {
                break;
            }
            continue;
        }
        for (DimType j = 0; j < tns->shape[1]; ++j) {
            if (ndims == 3) {
                size_t cb = min(rem, (size_t)tns->shape[2] * tns->type_bytes());
                gpu_memcpy(buf->ref(tns->offset_bytes(i, j, 0)), &ps[done], cb);
                rem -= cb;
                done += cb;
                if (rem == 0) {
                    break;
                }
                continue;
            }
            for (DimType k = 0; k < tns->shape[2]; ++k) {
                size_t cb = min(rem, (size_t)tns->shape[3] * tns->type_bytes());
                gpu_memcpy(buf->ref(tns->offset_bytes(i, j, k, 0)), &ps[done],
                           cb);
                rem -= cb;
                done += cb;
                if (rem == 0) {
                    break;
                }
            }
        }
    }
    assert(rem == 0);
    assert(done == bytes);
}

// Copy (possibly non-contiguous) data from a tensor on GPU to a contiguous
// host buffer. The given number of bytes is copied, in order of appearance
// on the memory. This function assumes that `dst` is large enough to hold
// the data.
void Executor::tensor_memcpy(void *dst, Tensor *src, size_t bytes)
{
    GpuBuf *buf = this->get_gpu_buf(src);
    if (buf == nullptr) {
        LOGERR("failed to get GPU buffer for tensor ", src->id);
    }
    Tensor *tns = this->get_tensor(src);
    if (bytes == 0) {
        bytes = tns->shape_bytes();
    } else if (bytes > (size_t)tns->shape_bytes()) {
        LOGERR("the given number of bytes (", bytes,
               ") is larger than the tensor size (", tns->shape_bytes(), ")");
    }
    int ndims = tns->ndims();
    char *pd = (char *)dst;
    if (ndims == 1) {
        gpu_memcpy(pd, buf->ref(tns->offset_bytes(0)), bytes);
        return;
    }
    size_t done = 0;
    size_t rem = bytes;
    for (DimType i = 0; i < tns->shape[0]; ++i) {
        if (ndims == 2) {
            size_t cb = min(rem, (size_t)tns->shape[1] * tns->type_bytes());
            gpu_memcpy(&pd[done], buf->ref(tns->offset_bytes(i, 0)), cb);
            rem -= cb;
            done += cb;
            if (rem == 0) {
                break;
            }
            continue;
        }
        for (DimType j = 0; j < tns->shape[1]; ++j) {
            if (ndims == 3) {
                size_t cb = min(rem, (size_t)tns->shape[2] * tns->type_bytes());
                gpu_memcpy(&pd[done], buf->ref(tns->offset_bytes(i, j, 0)), cb);
                rem -= cb;
                done += cb;
                if (rem == 0) {
                    break;
                }
                continue;
            }
            for (DimType k = 0; k < tns->shape[2]; ++k) {
                size_t cb = min(rem, (size_t)tns->shape[3] * tns->type_bytes());
                gpu_memcpy(&pd[done], buf->ref(tns->offset_bytes(i, j, k, 0)),
                           cb);
                rem -= cb;
                done += cb;
                if (rem == 0) {
                    break;
                }
            }
        }
    }
    assert(rem == 0);
    assert(done == bytes);
}

// Set all bytes of `tns` into zero.
void Executor::tensor_clear(Tensor *tns)
{
    GpuBuf *buf = this->get_gpu_buf(tns);
    if (buf == nullptr) {
        LOGERR("failed to get GPU buffer for tensor ", tns->id);
    }
    Tensor *_tns = this->get_tensor(tns);
    int ndims = _tns->ndims();
    size_t bytes = _tns->shape_bytes();
    assert(bytes % 4 == 0);
    size_t num = bytes >> 2;
    if (ndims == 1) {
        gpu_memset(buf->ref(_tns->offset_bytes(0)), 0, num);
        return;
    }
    size_t done = 0;
    size_t rem = num;
    for (DimType i = 0; i < _tns->shape[0]; ++i) {
        if (ndims == 2) {
            bytes = (size_t)_tns->shape[1] * _tns->type_bytes();
            assert(bytes % 4 == 0);
            size_t cn = min(rem, bytes >> 2);
            gpu_memset(buf->ref(_tns->offset_bytes(i, 0)), 0, cn);
            rem -= cn;
            done += cn;
            if (rem == 0) {
                break;
            }
            continue;
        }
        for (DimType j = 0; j < _tns->shape[1]; ++j) {
            if (ndims == 3) {
                bytes = (size_t)_tns->shape[2] * _tns->type_bytes();
                assert(bytes % 4 == 0);
                size_t cn = min(rem, bytes >> 2);
                gpu_memset(buf->ref(_tns->offset_bytes(i, j, 0)), 0, cn);
                rem -= cn;
                done += cn;
                if (rem == 0) {
                    break;
                }
                continue;
            }
            for (DimType k = 0; k < _tns->shape[2]; ++k) {
                bytes = (size_t)_tns->shape[3] * _tns->type_bytes();
                assert(bytes % 4 == 0);
                size_t cn = min(rem, bytes >> 2);
                gpu_memset(buf->ref(_tns->offset_bytes(i, j, k, 0)), 0, cn);
                rem -= cn;
                done += cn;
                if (rem == 0) {
                    break;
                }
            }
        }
    }
    assert(rem == 0);
    assert(done == num);
}

} // namespace ark
