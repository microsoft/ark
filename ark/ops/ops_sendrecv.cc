// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include <cassert>

#include "gpu/gpu_buf.h"
#include "logging.h"
#include "model.h"

namespace ark {

extern const OpConfigMap ConfigMap;

SendOp::SendOp(const std::string &prec_type, Tensor *input, Tensor *recvbuf,
               int sid, int rank, int dst_rank, size_t bytes,
               const std::string &name)
    : Op{OP_SEND,
         prec_type,
         {input, recvbuf},
         {input},
         {{rank, dst_rank, bytes, sid}},
         name,
         &ConfigMap,
         -1,
         true} {}

std::string SendOp::function_name(const OpConfig &) const {
    Tensor *input = this->inputs[0];
    CHECK(input->is_sequential());

    int rank;
    int dst_rank;
    size_t bytes;
    this->args.get(&rank, 0);
    this->args.get(&dst_rank, 1);
    this->args.get(&bytes, 2);

    return Op::function_name("ark::comm::send",
                             {{
                                 rank,      // Rank
                                 dst_rank,  // DstRank
                                 bytes,     // Length
                             }});
}

OpArgs SendOp::function_call_args(const OpConfig &) const {
    Tensor *input = this->inputs[0];
    Tensor *recvbuf = this->inputs[1];

    CHECK(input->buf != nullptr);
    CHECK(recvbuf->buf != nullptr);

    OpArgs opargs;
    // send(dst_offset, src_offset...)
    opargs.put((int)(recvbuf->buf->get_buf_offset() + recvbuf->offset_bytes()));
    opargs.put((int)(input->buf->get_buf_offset() + input->offset_bytes()));
    return opargs;
}

SendDoneOp::SendDoneOp(const std::string &prec_type, Tensor *input, int rank,
                       int dst_rank, const std::string &name)
    : Op{OP_SEND_DONE, prec_type,  {input}, {input}, {{rank, dst_rank}},
         name,         &ConfigMap, -1,      true} {}

std::string SendDoneOp::function_name(const OpConfig &) const {
    Tensor *input = this->inputs[0];
    CHECK(input->is_sequential());

    int rank;
    int dst_rank;
    this->args.get(&rank, 0);
    this->args.get(&dst_rank, 1);

    return Op::function_name("ark::comm::send_done",
                             {{
                                 rank,      // Rank
                                 dst_rank,  // DstRank
                             }});
}

OpArgs SendDoneOp::function_call_args(const OpConfig &) const { return {}; }

RecvOp::RecvOp(const std::string &prec_type, Tensor *output, int sid, int rank,
               int src_rank, size_t bytes, const std::string &name)
    : Op{OP_RECV, prec_type,  {}, {output}, {{rank, src_rank, bytes, sid}},
         name,    &ConfigMap, -1, true} {}

std::string RecvOp::function_name(const OpConfig &) const {
    Tensor *output = this->outputs[0];
    CHECK(output->is_sequential());

    int rank;
    int src_rank;
    this->args.get(&rank, 0);
    this->args.get(&src_rank, 1);

    return Op::function_name("ark::comm::recv",
                             {{
                                 rank,      // Rank
                                 src_rank,  // SrcRank
                             }});
}

OpArgs RecvOp::function_call_args(const OpConfig &) const { return {}; }

Tensor *Model::send(Tensor *input, int sid, int dst_rank, std::size_t bytes,
                    const std::string &name) {
    size_t max_bytes = input->ldims_bytes();
    if (max_bytes < bytes) {
        LOG(ERROR, "invalid bytes: ", bytes, ", max: ", max_bytes);
    }
    if (bytes == 0) {
        bytes = max_bytes;
    }
    LOG(DEBUG, "send ", input->shape, " ", dst_rank, " ", bytes);
    input->exported = true;

    Tensor *recvbuf = this->tensor(input->shape, input->type);
    recvbuf->imported_rank = dst_rank;
    SendOp op{"none",           input,    recvbuf, sid,
              this->impl->rank, dst_rank, bytes,   name};
    return this->impl->add_op(op)[0];
}

Tensor *Model::send_done(Tensor *input, int, int dst_rank,
                         const std::string &name) {
    LOG(DEBUG, "send_done ", input->shape, " ", dst_rank);
    SendDoneOp op{"none", input, this->impl->rank, dst_rank, name};
    return this->impl->add_op(op)[0];
}

Tensor *Model::recv(int sid, int src_rank, size_t bytes, Tensor *output,
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
    RecvOp op{"none", output, sid, this->impl->rank, src_rank, bytes, name};
    return this->impl->add_op(op)[0];
}

const OpConfigMap ConfigMap = {
    {{OP_ARCH_ANY, "none"},
     {
         // NumWarps, SmemBytes, InDepsTiles, OutDepsTiles, SyncPre, SyncPost
         {1, 0, {{-1, -1}, {-1, -1}}, {{-1, -1}}, true, true},
     }},
};

}  // namespace ark
