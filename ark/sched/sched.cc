// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "sched/sched.h"
#include "logging.h"

using namespace std;

namespace ark {

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

} // namespace ark