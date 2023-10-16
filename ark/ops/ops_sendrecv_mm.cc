// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include <cassert>

#include "env.h"
#include "logging.h"
#include "model.h"

namespace ark {

extern const OpConfigMap SendRecvMMConfigMap;

SendMMOp::SendMMOp(const std::string &prec_type, Tensor *input, Tensor *recvbuf,
                   Tensor *send_ready_flag, Tensor *output, int id, int gpu_dst,
                   size_t bytes, const std::string &name)
    : Op{OP_SEND_MM,
         prec_type,
         {input, recvbuf, send_ready_flag},
         {output},
         {{id, gpu_dst, bytes}},
         name,
         &SendRecvMMConfigMap,
         -1,
         true} {}

std::string SendMMOp::function_name(const OpConfig &cfg) const {
    Tensor *input = this->inputs[0];
    Dims shp_in = input->shape;
    Tensor *output = this->outputs[0];

    int ndims = output->shape.ndims();
    const OpTile &tile_out = cfg.output_tiles[0];
    CHECK(output->ldims[ndims - 1] % tile_out.y == 0);
    if (ndims > 1) {
        CHECK(output->ldims[ndims - 2] % tile_out.x == 0);
    } else {
        CHECK(tile_out.x == 1);
    }

    CHECK(ndims == 2);
    DimType m = shp_in[ndims - 1];
    DimType n = shp_in[ndims - 2];
    Dims pad_in = input->pads;
    const OpTile &tile_in = cfg.input_tiles[0];
    // Verify paddings
    CHECK((shp_in[ndims - 2] % tile_in.x == 0) ||
          (pad_in[ndims - 2] >= tile_in.x));
    CHECK((shp_in[ndims - 1] % tile_in.y == 0) ||
          (pad_in[ndims - 1] >= tile_in.y));

    return Op::function_name("ark::comm::sendLL",
                             {{
                                 m,                   // LDM
                                 n,                   // LDN
                                 cfg.num_warps * 32,  // TN
                                 cfg.smem_bytes,      // SmemBytes
                                 tile_in.y,           // TDM
                                 tile_in.x,           // TDN
                                 1,                   // FLAG
                             }});
}

OpArgs SendMMOp::function_call_args(const OpConfig &) const {
    OpArgs opargs;
    opargs.put(this->inputs[1]);
    opargs.put(this->inputs[0]);
    opargs.put(this->inputs[2]);
    return opargs;
}

RecvMMOp::RecvMMOp(const std::string &prec_type, Tensor *input, Tensor *recvbuf,
                   Tensor *send_ready_flag, Tensor *output, int id, int gpu_src,
                   size_t bytes, const std::string &name)
    : Op{OP_RECV_MM,
         prec_type,
         {input, recvbuf, send_ready_flag},
         {output},
         {{id, gpu_src, bytes}},
         name,
         &SendRecvMMConfigMap,
         -1,
         true} {}

std::string RecvMMOp::function_name(const OpConfig &cfg) const {
    Tensor *input = this->inputs[0];
    Dims shp_in = input->shape;
    Tensor *output = this->outputs[0];

    int ndims = output->shape.ndims();
    const OpTile &tile_out = cfg.output_tiles[0];
    CHECK(output->ldims[ndims - 1] % tile_out.y == 0);
    if (ndims > 1) {
        CHECK(output->ldims[ndims - 2] % tile_out.x == 0);
    } else {
        CHECK(tile_out.x == 1);
    }

    CHECK(ndims == 2);
    DimType m = shp_in[ndims - 1];
    DimType n = shp_in[ndims - 2];
    Dims pad_in = input->pads;
    const OpTile &tile_in = cfg.input_tiles[0];
    // Verify paddings
    CHECK((shp_in[ndims - 2] % tile_in.x == 0) ||
          (pad_in[ndims - 2] >= tile_in.x));
    CHECK((shp_in[ndims - 1] % tile_in.y == 0) ||
          (pad_in[ndims - 1] >= tile_in.y));

    return Op::function_name("ark::comm::recvLL",
                             {{
                                 m,                   // LDM
                                 n,                   // LDN
                                 cfg.num_warps * 32,  // TN
                                 cfg.smem_bytes,      // SmemBytes
                                 tile_in.y,           // TDM
                                 tile_in.x,           // TDN
                                 1,                   // FLAG
                             }});
}

OpArgs RecvMMOp::function_call_args(const OpConfig &) const {
    OpArgs opargs;
    opargs.put(this->inputs[1]);
    opargs.put(this->inputs[0]);
    opargs.put(this->inputs[2]);
    return opargs;
}

// TODO: set the max_tile_num according to the tile number of the op
const int max_tile_num = 2048;

// send data from src to dst of id
Tensor *Model::send_mm(Tensor *input, int id, int gpu_dst, size_t bytes,
                       Tensor *output, const std::string &name) {
    assert(input != nullptr);
    size_t max_bytes = input->ldims_bytes();
    if (max_bytes < bytes) {
        LOG(ERROR, "invalid bytes: ", bytes, ", max: ", max_bytes);
    }
    if (bytes == 0) {
        bytes = max_bytes;
    }
    if (output != nullptr && input->type != output->type) {
        LOG(ERROR, "invalid output data type: ", output->type);
    }
    if (output == nullptr) {
        output = this->tensor(input->shape, input->type, input->buf);
    } else if (output->shape != input->shape) {
        LOG(ERROR, "invalid output shape: ", output->shape);
    }
    Dims recvbuf_shape = input->shape;
    int ndims = recvbuf_shape.ndims();
    CHECK(ndims == 2);
    recvbuf_shape[ndims - 2] *= 2 * input->type_bytes();
    Tensor *recvbuf = this->tensor(recvbuf_shape, ark::BYTE);
    recvbuf->imported_rank = gpu_dst;
    Tensor *send_ready_flag = this->tensor(
        {
            max_tile_num,
        },
        INT32);
    send_ready_flag->exported = true;
    SendMMOp op{"none",  input, recvbuf, send_ready_flag, output, id,
                gpu_dst, bytes, name};
    return this->impl->add_op(op)[0];
}

//
Tensor *Model::recv_mm(Tensor *input, int id, int gpu_src, size_t bytes,
                       Tensor *output, const std::string &name) {
    assert(input != nullptr);
    size_t max_bytes = input->ldims_bytes();
    if (max_bytes < bytes) {
        LOG(ERROR, "invalid bytes: ", bytes, ", max: ", max_bytes);
    }
    if (bytes == 0) {
        bytes = max_bytes;
    }
    input->exported = true;

    if (output != nullptr && input->type != output->type) {
        LOG(ERROR, "invalid output data type: ", output->type);
    }
    if (output == nullptr) {
        output = this->tensor(input->shape, input->type, input->buf);
    } else if (output->shape != input->shape) {
        LOG(ERROR, "invalid output shape: ", output->shape);
    }
    // use a tensor as recvbuf to store the received data, the size of the
    // recvbuf is twice of the input
    Dims recvbuf_shape = input->shape;
    int ndims = recvbuf_shape.ndims();
    CHECK(ndims == 2);
    recvbuf_shape[ndims - 2] *= 2 * input->type_bytes();
    Tensor *recvbuf = this->tensor(recvbuf_shape, ark::BYTE);
    recvbuf->exported = true;
    Tensor *send_ready_flag = this->tensor(
        {
            max_tile_num,
        },
        INT32);
    send_ready_flag->imported_rank = gpu_src;
    RecvMMOp op{"none",  input, recvbuf, send_ready_flag, output, id,
                gpu_src, bytes, name};
    return this->impl->add_op(op)[0];
}

const OpConfigMap SendRecvMMConfigMap = {
    {{OP_ARCH_ANY, "none"},
     {
         // NumWarps, SmemBytes, InDepsTiles, OutDepsTiles, SyncPre, SyncPost
         {4, 0, {{64, 64}, {64, 64}, {1, 1}}, {{64, 64}}, false, false},
         {2, 0, {{32, 64}, {32, 64}, {1, 1}}, {{32, 64}}, false, false},
         {1, 0, {{16, 64}, {16, 64}, {1, 1}}, {{16, 64}}, false, false},
         {1, 0, {{8, 64}, {8, 64}, {1, 1}}, {{8, 64}}, false, false},
         {1, 0, {{2, 128}, {2, 128}, {1, 1}}, {{2, 128}}, false, false},
         {1, 0, {{4, 64}, {4, 64}, {1, 1}}, {{4, 64}}, false, false},
         {1, 0, {{2, 64}, {2, 64}, {1, 1}}, {{2, 64}}, false, false},
     }},
};

}  // namespace ark
