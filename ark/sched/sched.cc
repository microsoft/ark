// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "sched/sched.h"
#include "logging.h"

using namespace std;

namespace ark {

BaseScheduler::BaseScheduler(Model &model, int gpu_id, int rank_,
                             int world_size_, int num_warps_per_sm_)
    : model{&model}, gpu_mgr{get_gpu_mgr(gpu_id)}, rank{rank_}, world_size{
                                                                    world_size_}
{
    const GpuInfo &gpu_info = this->gpu_mgr->get_gpu_info();
    int min_wps = gpu_info.min_threads_per_block / gpu_info.threads_per_warp;
    this->num_warps_per_sm = std::max(num_warps_per_sm_, min_wps);
}

// create context on gpu for the model
GpuMgrCtx *BaseScheduler::create_context(const std::string &name)
{
    GpuMgrCtx *ctx =
        this->gpu_mgr->create_context(name, this->rank, this->world_size);
    for (BufInfo &bi : this->buf_infos) {
        GpuBuf *buf;
        if (bi.gpu_id == this->gpu_mgr->gpu_id) {
            auto search = this->buf_trans.find(bi.tbuf);
            if (search != this->buf_trans.end()) {
                // Already allocated.
                buf = search->second;
                if (bi.sid != -1) {
                    ctx->mem_export(this->buf_trans[bi.tbuf], bi.offset,
                                    bi.sid);
                }
            } else if (bi.sid == -1) {
                buf = ctx->mem_alloc(bi.bytes, 1);
            } else {
                // Align for RDMA performance.
                buf = ctx->mem_alloc(bi.bytes, 65536);
                ctx->mem_export(buf, bi.offset, bi.sid);
            }
        } else {
            buf = ctx->mem_import(bi.bytes, bi.sid, bi.gpu_id);
        }
        this->buf_trans[bi.tbuf] = buf;
    }
    for (auto &srop : this->send_recv_ops) {
        int sid;
        int remote_rank;
        size_t bytes;
        srop->args.get(&sid, 0);
        srop->args.get(&remote_rank, 2);
        srop->args.get(&bytes, 3);

        LOG(DEBUG, "reg_sendrecv: sid=", sid, " remote=", remote_rank,
            " bytes=", bytes, " is_recv=", srop->type == OP_RECV);
        ctx->reg_sendrecv(sid, remote_rank, bytes, srop->type == OP_RECV);
    }
    ctx->freeze();
    this->ctx = ctx;
    return ctx;
}

GpuBuf *BaseScheduler::get_gpu_buf(Tensor *tns) const
{
    if (tns == nullptr) {
        return nullptr;
    }
    if (tns->buf == nullptr) {
        return nullptr;
    }
    auto search = this->buf_trans.find(tns->buf);
    if (search == this->buf_trans.end()) {
        return nullptr;
    }
    return search->second;
}

} // namespace ark