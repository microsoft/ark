// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include <cassert>

#include "gpu/gpu_buf.h"
#include "logging.h"
#include "model.h"

namespace ark {

extern const OpConfigMap MsllConfigMap;

MsllSendOp::MsllSendOp(const std::string &prec_type, Tensor *input,
                       Tensor *recvbuf, int sid, int rank, int dst_rank,
                       size_t bytes, const std::string &name)
    : Op{OP_SEND_MSLL,
         prec_type,
         {input, recvbuf},
         {input},
         {{rank, dst_rank, bytes, sid}},
         name,
         &MsllConfigMap,
         -1,
         true} {}

std::string MsllSendOp::function_name(const OpConfig &) const {
    Tensor *input = this->inputs[0];
    CHECK(input->is_sequential());

    int rank;
    int dst_rank;
    size_t bytes;
    this->args.get(&rank, 0);
    this->args.get(&dst_rank, 1);
    this->args.get(&bytes, 2);

    return Op::function_name("ark::comm::send_msll",
                             {{
                                 rank,      // Rank
                                 dst_rank,  // DstRank
                                 bytes,     // Length
                             }});
}

OpArgs MsllSendOp::function_call_args(const OpConfig &) const {
    Tensor *input = this->inputs[0];
    Tensor *recvbuf = this->inputs[1];

    CHECK(input->buf != nullptr);
    CHECK(recvbuf->buf != nullptr);

    OpArgs opargs;
    // send_msll(dst_offset, src_offset...)
    opargs.put((int)(recvbuf->buf->get_buf_offset() + recvbuf->offset_bytes()));
    opargs.put((int)(input->buf->get_buf_offset() + input->offset_bytes()));
    return opargs;
}

MsllSendDoneOp::MsllSendDoneOp(const std::string &prec_type, Tensor *input,
                               int rank, int dst_rank, const std::string &name)
    : Op{OP_SEND_DONE_MSLL,
         prec_type,
         {input},
         {input},
         {{rank, dst_rank}},
         name,
         &MsllConfigMap,
         -1,
         true} {}

std::string MsllSendDoneOp::function_name(const OpConfig &) const {
    Tensor *input = this->inputs[0];
    CHECK(input->is_sequential());

    int rank;
    int dst_rank;
    this->args.get(&rank, 0);
    this->args.get(&dst_rank, 1);

    return Op::function_name("ark::comm::send_done_msll",
                             {{
                                 rank,      // Rank
                                 dst_rank,  // DstRank
                             }});
}

OpArgs MsllSendDoneOp::function_call_args(const OpConfig &) const { return {}; }

MsllRecvOp::MsllRecvOp(const std::string &prec_type, Tensor *output, int sid,
                       int rank, int src_rank, size_t bytes,
                       const std::string &name)
    : Op{OP_RECV_MSLL,
         prec_type,
         {},
         {output},
         {{rank, src_rank, bytes, sid}},
         name,
         &MsllConfigMap,
         -1,
         true} {}

std::string MsllRecvOp::function_name(const OpConfig &) const {
    Tensor *output = this->outputs[0];
    CHECK(output->is_sequential());

    int rank;
    int src_rank;
    this->args.get(&rank, 0);
    this->args.get(&src_rank, 1);

    return Op::function_name("ark::comm::recv_msll",
                             {{
                                 rank,      // Rank
                                 src_rank,  // SrcRank
                             }});
}

OpArgs MsllRecvOp::function_call_args(const OpConfig &) const { return {}; }

Tensor *Model::send_msll(Tensor *input, int sid, int dst_rank,
                         std::size_t bytes, const std::string &name) {
    size_t max_bytes = input->ldims_bytes();
    if (max_bytes < bytes) {
        ERR(InvalidUsageError, "invalid bytes: ", bytes, ", max: ", max_bytes);
    }
    if (bytes == 0) {
        bytes = max_bytes;
    }
    LOG(DEBUG, "send_msll ", input->shape, " ", dst_rank, " ", bytes);
    input->exported = true;

    Tensor *recvbuf = this->tensor(input->shape, input->type);
    recvbuf->imported_rank = dst_rank;
    MsllSendOp op{"none",           input,    recvbuf, sid,
                  this->impl->rank, dst_rank, bytes,   name};
    return this->impl->add_op(op)[0];
}

Tensor *Model::send_done_msll(Tensor *input, int dst_rank,
                              const std::string &name) {
    LOG(DEBUG, "send_done_msll ", input->shape, " ", dst_rank);
    MsllSendDoneOp op{"none", input, this->impl->rank, dst_rank, name};
    return this->impl->add_op(op)[0];
}

Tensor *Model::recv_msll(int sid, int src_rank, size_t bytes, Tensor *output,
                         const std::string &name) {
    if (output == nullptr) {
        if (bytes == 0) {
            ERR(InvalidUsageError, "receive bytes cannot be 0");
        }
        output = this->tensor({DimType(bytes)}, BYTE);
    }
    output->exported = true;
    size_t max_bytes = output->shape_bytes();
    if (max_bytes < bytes) {
        ERR(InvalidUsageError, "invalid bytes: ", bytes, ", max: ", max_bytes);
    }
    if (bytes == 0) {
        bytes = max_bytes;
    }
    MsllRecvOp op{"none", output, sid, this->impl->rank, src_rank, bytes, name};
    return this->impl->add_op(op)[0];
}

Tensor *Model::device_sync_msll(Tensor *input, int nranks,
                                const std::string &name) {
    MsllDeviceSyncOp op{"none", input, input, nranks, name};
    return this->impl->add_op(op)[0];
}

const OpConfigMap MsllConfigMap = {
    {{OP_ARCH_CUDA_ANY, "none"},
     {
         // NumWarps, SmemBytes, InDepsTiles, OutDepsTiles, SyncPre, SyncPost
         {1, 0, {{-1, -1}, {-1, -1}}, {{-1, -1}}, true, true},
     }},
};

}  // namespace ark
