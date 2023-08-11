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

// Get the corresponding GPU buffer of the executor from the given model tensor.
GpuBuf *Executor::get_gpu_buf(Tensor *tns) const
{
    return this->impl->sched->get_gpu_buf(tns);
}

// Copy contiguous data from a host buffer to the given tensor's (possibly
// non-contiguous) data range on GPU.
void Executor::tensor_memcpy(Tensor *dst, const void *src, size_t bytes)
{
    GpuBuf *buf = this->get_gpu_buf(dst);
    if (buf == nullptr) {
        LOGERR("failed to get GPU buffer for tensor ", dst->id);
    }
    Tensor *tns = dst;
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
    Tensor *tns = src;
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
    int ndims = tns->ndims();
    size_t bytes = tns->shape_bytes();
    assert(bytes % 4 == 0);
    size_t num = bytes >> 2;
    if (ndims == 1) {
        gpu_memset(buf->ref(tns->offset_bytes(0)), 0, num);
        return;
    }
    size_t done = 0;
    size_t rem = num;
    for (DimType i = 0; i < tns->shape[0]; ++i) {
        if (ndims == 2) {
            bytes = (size_t)tns->shape[1] * tns->type_bytes();
            assert(bytes % 4 == 0);
            size_t cn = min(rem, bytes >> 2);
            gpu_memset(buf->ref(tns->offset_bytes(i, 0)), 0, cn);
            rem -= cn;
            done += cn;
            if (rem == 0) {
                break;
            }
            continue;
        }
        for (DimType j = 0; j < tns->shape[1]; ++j) {
            if (ndims == 3) {
                bytes = (size_t)tns->shape[2] * tns->type_bytes();
                assert(bytes % 4 == 0);
                size_t cn = min(rem, bytes >> 2);
                gpu_memset(buf->ref(tns->offset_bytes(i, j, 0)), 0, cn);
                rem -= cn;
                done += cn;
                if (rem == 0) {
                    break;
                }
                continue;
            }
            for (DimType k = 0; k < tns->shape[2]; ++k) {
                bytes = (size_t)tns->shape[3] * tns->type_bytes();
                assert(bytes % 4 == 0);
                size_t cn = min(rem, bytes >> 2);
                gpu_memset(buf->ref(tns->offset_bytes(i, j, k, 0)), 0, cn);
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

void Executor::print_tensor(Tensor *tns)
{
    half_t *p = (half_t *)malloc(tns->shape_bytes());
    this->tensor_memcpy(p, tns, tns->shape_bytes());
    for (DimType i = 0; i < tns->shape[0]; ++i) {
        for (DimType j = 0; j < tns->shape[1]; ++j) {
            for (DimType k = 0; k < tns->shape[2]; ++k) {
                for (DimType l = 0; l < tns->shape[3]; ++l) {
                    printf("%f ", (float)p[tns->offset(i, j, k, l)]);
                }
                printf("\n");
            }
            printf("\n");
        }
        printf("\n");
    }
}

} // namespace ark
