// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "logging.h"
#include "model.h"

using namespace std;

namespace ark {

class SendOp : public Op
{
  public:
    SendOp::SendOp(OpPrecType prec_type, Tensor *input, Tensor *output,
                   const string &name);
};

class SendDoneOp : public Op
{
  public:
    SendDoneOp::SendDoneOp(OpPrecType prec_type, Tensor *input, Tensor *output,
                           const string &name);
};

class RecvOp : public Op
{
  public:
    RecvOp::RecvOp(OpPrecType prec_type, Tensor *input, Tensor *output,
                   const string &name);
};

SendOp::SendOp(OpPrecType prec_type, Tensor *input, Tensor *output,
               const string &name)
    : Op{OP_SEND, prec_type, {input}, {output}, {}, name, -1}
{
}

SendDoneOp::SendDoneOp(OpPrecType prec_type, Tensor *input, Tensor *output,
                       const string &name)
    : Op{OP_SEND_DONE, prec_type, {input}, {output}, {}, name, -1}
{
}

RecvOp::RecvOp(OpPrecType prec_type, Tensor *input, Tensor *output,
               const string &name)
    : Op{OP_RECV, prec_type, {input}, {output}, {}, name, -1}
{
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
    SendOp op{OP_PREC_NONE, input, output, name};
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
    SendDoneOp op{OP_PREC_NONE, input, output, name};
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
    RecvOp op{OP_PREC_NONE, input, output, name};
    this->impl->add_op(op);
    return output;
}

} // namespace ark
