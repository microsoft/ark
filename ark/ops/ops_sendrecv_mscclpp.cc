// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include <cassert>

#include "gpu/gpu_buf.h"
#include "logging.h"
#include "model.h"

namespace ark {

extern const OpConfigMap MscclppConfigMap;
extern const OpConfigMap MscclppSyncConfigMap;

MscclppSendOp::MscclppSendOp(const std::string &prec_type, Tensor *input,
                             Tensor *recvbuf, int sid, int rank, int dst_rank,
                             size_t bytes, const std::string &name)
    : Op{OP_SEND_MSCCLPP,
         prec_type,
         {input, recvbuf},
         {input},
         {{rank, dst_rank, bytes, sid}},
         name,
         &MscclppConfigMap,
         -1,
         true} {}

std::string MscclppSendOp::function_name(const OpConfig &) const {
    Tensor *input = this->inputs[0];
    CHECK(input->is_sequential());

    int rank;
    int dst_rank;
    size_t bytes;
    this->args.get(&rank, 0);
    this->args.get(&dst_rank, 1);
    this->args.get(&bytes, 2);

    return Op::function_name("ark::comm::send_mscclpp",
                             {{
                                 rank,      // Rank
                                 dst_rank,  // DstRank
                                 bytes,     // Length
                             }});
}

OpArgs MscclppSendOp::function_call_args(const OpConfig &) const {
    Tensor *input = this->inputs[0];
    Tensor *recvbuf = this->inputs[1];

    CHECK(input->buf != nullptr);
    CHECK(recvbuf->buf != nullptr);

    OpArgs opargs;
    // send_mscclpp(dst_offset, src_offset...)
    opargs.put((int)(recvbuf->buf->get_buf_offset() + recvbuf->offset_bytes()));
    opargs.put((int)(input->buf->get_buf_offset() + input->offset_bytes()));
    return opargs;
}

MscclppSendDoneOp::MscclppSendDoneOp(const std::string &prec_type,
                                     Tensor *input, int rank, int dst_rank,
                                     const std::string &name)
    : Op{OP_SEND_DONE_MSCCLPP,
         prec_type,
         {input},
         {input},
         {{rank, dst_rank}},
         name,
         &MscclppConfigMap,
         -1,
         true} {}

std::string MscclppSendDoneOp::function_name(const OpConfig &) const {
    Tensor *input = this->inputs[0];
    CHECK(input->is_sequential());

    int rank;
    int dst_rank;
    this->args.get(&rank, 0);
    this->args.get(&dst_rank, 1);

    return Op::function_name("ark::comm::send_done_mscclpp",
                             {{
                                 rank,      // Rank
                                 dst_rank,  // DstRank
                             }});
}

OpArgs MscclppSendDoneOp::function_call_args(const OpConfig &) const {
    return {};
}

MscclppRecvOp::MscclppRecvOp(const std::string &prec_type, Tensor *output,
                             int sid, int rank, int src_rank, size_t bytes,
                             const std::string &name)
    : Op{OP_RECV_MSCCLPP,
         prec_type,
         {},
         {output},
         {{rank, src_rank, bytes, sid}},
         name,
         &MscclppConfigMap,
         -1,
         true} {}

std::string MscclppRecvOp::function_name(const OpConfig &) const {
    Tensor *output = this->outputs[0];
    CHECK(output->is_sequential());

    int rank;
    int src_rank;
    this->args.get(&rank, 0);
    this->args.get(&src_rank, 1);

    return Op::function_name("ark::comm::recv_mscclpp",
                             {{
                                 rank,      // Rank
                                 src_rank,  // SrcRank
                             }});
}

OpArgs MscclppRecvOp::function_call_args(const OpConfig &) const { return {}; }

MscclppDeviceSyncOp::MscclppDeviceSyncOp(const std::string &prec_type,
                                         Tensor *input, Tensor *output,
                                         int nranks, const std::string &name)
    : Op{OP_DEVICE_SYNC_MSCCLPP, prec_type, {input}, {output}, {{nranks}}, name,
         &MscclppSyncConfigMap,  -1,        true} {}

std::string MscclppDeviceSyncOp::function_name(const OpConfig &) const {
    int nranks;
    this->args.get(&nranks, 0);
    return Op::function_name("ark::comm::device_sync_mscclpp", {{nranks}});
}

OpArgs MscclppDeviceSyncOp::function_call_args(const OpConfig &) const {
    return {};
}

Tensor *Model::send_mscclpp(Tensor *input, int sid, int dst_rank,
                            std::size_t bytes, const std::string &name) {
    size_t max_bytes = input->ldims_bytes();
    if (max_bytes < bytes) {
        LOG(ERROR, "invalid bytes: ", bytes, ", max: ", max_bytes);
    }
    if (bytes == 0) {
        bytes = max_bytes;
    }
    LOG(DEBUG, "send_mscclpp ", input->shape, " ", dst_rank, " ", bytes);
    input->exported = true;

    Tensor *recvbuf = this->tensor(input->shape, input->type);
    recvbuf->imported_rank = dst_rank;
    MscclppSendOp op{"none",           input,    recvbuf, sid,
                     this->impl->rank, dst_rank, bytes,   name};
    return this->impl->add_op(op)[0];
}

Tensor *Model::send_done_mscclpp(Tensor *input, int dst_rank,
                                 const std::string &name) {
    LOG(DEBUG, "send_done_mscclpp ", input->shape, " ", dst_rank);
    MscclppSendDoneOp op{"none", input, this->impl->rank, dst_rank, name};
    return this->impl->add_op(op)[0];
}

Tensor *Model::recv_mscclpp(int sid, int src_rank, size_t bytes, Tensor *output,
                            const std::string &name) {
    if (output == nullptr) {
        if (bytes == 0) {
            LOG(ERROR, "receive bytes cannot be 0");
        }
        output = this->tensor({DimType(bytes)}, BYTE);
    }
    output->exported = true;
    size_t max_bytes = output->shape_bytes();
    if (max_bytes < bytes) {
        LOG(ERROR, "invalid bytes: ", bytes, ", max: ", max_bytes);
    }
    if (bytes == 0) {
        bytes = max_bytes;
    }
    MscclppRecvOp op{"none",   output, sid, this->impl->rank,
                     src_rank, bytes,  name};
    return this->impl->add_op(op)[0];
}

Tensor *Model::device_sync_mscclpp(Tensor *input, int nranks,
                                   const std::string &name) {
    MscclppDeviceSyncOp op{"none", input, input, nranks, name};
    return this->impl->add_op(op)[0];
}

const OpConfigMap MscclppConfigMap = {
    {{OP_ARCH_CUDA_ANY, "none"},
     {
         // NumWarps, SmemBytes, InDepsTiles, OutDepsTiles, SyncPre, SyncPost
         {1, 0, {{-1, -1}}, {{-1, -1}}, true, true},
     }},
};

const OpConfigMap MscclppSyncConfigMap = {
    {{OP_ARCH_CUDA_ANY, "none"},
     {
         // NumWarps, SmemBytes, InDepsTiles, OutDepsTiles, SyncPre, SyncPost
         {1, 0, {{-1, -1}, {-1, -1}}, {{-1, -1}}, false, true},
     }},
};

}  // namespace ark
