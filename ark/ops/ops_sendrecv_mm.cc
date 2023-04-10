// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "ark/env.h"
#include "ark/logging.h"
#include "ark/model_io.h"

using namespace std;

namespace ark {

// TODO: set the max_tile_num according to the tile number of the op
const int max_tile_num = 2048;

// send data from src to dst of id
Tensor *Model::send_mm(Tensor *input, int id, int gpu_dst, size_t bytes,
                       Tensor *output, const string &name)
{
    assert(input != nullptr);
    size_t max_bytes = input->ldims_bytes();
    if (max_bytes < bytes) {
        LOGERR("invalid bytes: ", bytes, ", max: ", max_bytes);
    }
    if (bytes == 0) {
        bytes = max_bytes;
    }
    LOG(DEBUG, "send_mm", input->shape, " ", id, " ", gpu_dst, " ", bytes);
    if (output != nullptr && input->type != output->type) {
        LOGERR("invalid output data type: ", type_str(output->type));
    }
    if (output == nullptr) {
        output = this->tensor(input->shape, input->type, input->buf);
    } else if (output->shape != input->shape) {
        LOGERR("invalid output shape: ", output->shape);
    }
    Dims recvbuf_shape = input->shape;
    int ndims = recvbuf_shape.ndims();
    recvbuf_shape[ndims - 2] *= 2;
    Tensor *recvbuf = this->tensor(recvbuf_shape, input->type);
    recvbuf->imported = true;
    Tensor *send_ready_flag = this->tensor(
        {
            max_tile_num,
        },
        INT32);
    send_ready_flag->exported = true;
    this->create_op(OP_SEND_MM, OP_PREC_NONE, {input, recvbuf, send_ready_flag},
                    {output}, {id, gpu_dst, bytes}, name);

    return output;
}

//
Tensor *Model::recv_mm(Tensor *input, int id, int gpu_src, size_t bytes,
                       Tensor *output, const string &name)
{
    assert(input != nullptr);
    size_t max_bytes = input->ldims_bytes();
    if (max_bytes < bytes) {
        LOGERR("invalid bytes: ", bytes, ", max: ", max_bytes);
    }
    if (bytes == 0) {
        bytes = max_bytes;
    }
    LOG(DEBUG, "recv_mm", input->shape, " ", id, " ", gpu_src, " ", bytes);
    input->exported = true;

    if (output != nullptr && input->type != output->type) {
        LOGERR("invalid output data type: ", type_str(output->type));
    }
    if (output == nullptr) {
        output = this->tensor(input->shape, input->type, input->buf);
    } else if (output->shape != input->shape) {
        LOGERR("invalid output shape: ", output->shape);
    }
    // use a tensor as recvbuf to store the received data, the size of the
    // recvbuf is twice of the input
    Dims recvbuf_shape = input->shape;
    int ndims = recvbuf_shape.ndims();
    recvbuf_shape[ndims - 2] *= 2;
    Tensor *recvbuf = this->tensor(recvbuf_shape, input->type);
    recvbuf->exported = true;
    Tensor *send_ready_flag = this->tensor(
        {
            max_tile_num,
        },
        INT32);
    send_ready_flag->imported = true;
    this->create_op(OP_RECV_MM, OP_PREC_NONE, {input, recvbuf, send_ready_flag},
                    {output}, {id, gpu_src, bytes}, name);
    return output;
}

} // namespace ark
