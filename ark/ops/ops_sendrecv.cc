// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "logging.h"
#include "model.h"

using namespace std;

namespace ark {

SendOp::SendOp(OpPrecType prec_type, Tensor *input, Tensor *output, int sid,
               int rank, int dst_rank, size_t bytes, const string &name)
    : Op{OP_SEND, prec_type, {input}, {output}, {{sid, rank, dst_rank, bytes}},
         name,    -1,        true}
{
}

std::string SendOp::function_name(const OpConfig &) const
{
    Tensor *input = this->in_deps[0];
    CHECK(input->is_sequential());

    int sid;
    int rank;
    int dst_rank;
    size_t bytes;
    this->args.get(&sid, 0);
    this->args.get(&rank, 1);
    this->args.get(&dst_rank, 2);
    this->args.get(&bytes, 3);

    return Op::function_name("ark::comm::send", {{
                                                    rank,     // Rank
                                                    dst_rank, // DstRank
                                                    sid,      // SrcSid
                                                    sid,      // DstSid
                                                    bytes,    // Length
                                                }});
}

OpArgs SendOp::function_call_args(const OpConfig &cfg) const
{
    return {};
}

SendDoneOp::SendDoneOp(OpPrecType prec_type, Tensor *input, Tensor *output,
                       int sid, int rank, int dst_rank, const string &name)
    : Op{OP_SEND_DONE, prec_type, {input}, {output}, {{sid, rank, dst_rank}},
         name,         -1,        true}
{
}

std::string SendDoneOp::function_name(const OpConfig &) const
{
    int sid;
    int rank;
    int dst_rank;
    this->args.get(&sid, 0);
    this->args.get(&rank, 1);
    this->args.get(&dst_rank, 2);

    return Op::function_name("ark::comm::send_done", {{
                                                         rank,     // Rank
                                                         dst_rank, // DstRank
                                                         sid,      // SrcSid
                                                     }});
}

OpArgs SendDoneOp::function_call_args(const OpConfig &cfg) const
{
    return {};
}

RecvOp::RecvOp(OpPrecType prec_type, Tensor *input, Tensor *output, int sid,
               int rank, int src_rank, size_t bytes, const string &name)
    : Op{OP_RECV, prec_type, {input}, {output}, {{sid, rank, src_rank, bytes}},
         name,    -1,        true}
{
}

std::string RecvOp::function_name(const OpConfig &) const
{
    Tensor *input = this->in_deps[0];
    CHECK(input->is_sequential());

    int sid;
    int rank;
    int src_rank;
    this->args.get(&sid, 0);
    this->args.get(&rank, 1);
    this->args.get(&src_rank, 2);

    return Op::function_name("ark::comm::recv", {{
                                                    rank,     // Rank
                                                    src_rank, // DstRank
                                                    sid,      // SrcSid
                                                }});
}

OpArgs RecvOp::function_call_args(const OpConfig &cfg) const
{
    return {};
}

//
Tensor *Model::send(Tensor *input, int id, int dst_rank, size_t bytes,
                    Tensor *output, const string &name)
{
    size_t max_bytes = input->shape_bytes();
    if (max_bytes < bytes) {
        LOGERR("invalid bytes: ", bytes, ", max: ", max_bytes);
    }
    if (bytes == 0) {
        bytes = max_bytes;
    }
    LOG(DEBUG, "send ", input->shape, " ", id, " ", dst_rank, " ", bytes);
    input->exported = true;
    if (output == nullptr) {
        output = this->tensor({1, 1, 1, 1}, INT32);
    }
    SendOp op{OP_PREC_NONE,     input,    output, id,
              this->impl->rank, dst_rank, bytes,  name};
    this->impl->add_op(op);
    return output;
}

//
Tensor *Model::send_done(Tensor *input, int id, int dst_rank, Tensor *output,
                         const string &name)
{
    LOG(DEBUG, "send_done ", input->shape, " ", id);
    if (output == nullptr) {
        output = this->tensor({1, 1, 1, 1}, INT32);
    }
    SendDoneOp op{OP_PREC_NONE,     input,    output, id,
                  this->impl->rank, dst_rank, name};
    this->impl->add_op(op);
    return output;
}

//
Tensor *Model::recv(Tensor *input, int id, int src_rank, size_t bytes,
                    Tensor *output, const string &name)
{
    assert(input != nullptr);
    size_t max_bytes = input->shape_bytes();
    if (max_bytes < bytes) {
        LOGERR("invalid bytes: ", bytes, ", max: ", max_bytes);
    }
    if (bytes == 0) {
        bytes = max_bytes;
    }
    LOG(DEBUG, "recv ", input->shape, " ", id, " ", src_rank, " ", bytes);
    input->exported = true;
    if (output == nullptr) {
        output = this->tensor({1, 1, 1, 1}, INT32);
    }
    RecvOp op{OP_PREC_NONE,     input,    output, id,
              this->impl->rank, src_rank, bytes,  name};
    this->impl->add_op(op);
    return output;
}

} // namespace ark
