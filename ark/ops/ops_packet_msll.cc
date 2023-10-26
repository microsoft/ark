// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include <cassert>

#include "env.h"
#include "logging.h"
#include "model.h"

#ifdef ARK_USE_MSLL
#include <msll/packet.hpp>
#endif

constexpr int MSLL_PACKET_SIZE = 16;

namespace ark {

extern const OpConfigMap MsllPacketConfigMap;

MsllPutPacketOp::MsllPutPacketOp(const std::string &prec_type, Tensor *input,
                                 Tensor *local_tmp_buf, Tensor *recv_buf,
                                 int id, int rank, int dst_rank,
                                 size_t dst_offset, int flag,
                                 const std::string &name)
    : Op{OP_PUT_PACKET_MSLL,
         prec_type,
         {input, local_tmp_buf, recv_buf},
         {input},
         {{id, rank, dst_rank, dst_offset, flag}},
         name,
         &MsllPacketConfigMap,
         -1,
         true} {}

std::string MsllPutPacketOp::function_name(const OpConfig &cfg) const {
    Tensor *input = this->inputs[0];

    int rank;
    int dst_rank;
    size_t dst_offset;
    int flag;

    this->args.get(&rank, 1);
    this->args.get(&dst_rank, 2);
    this->args.get(&dst_offset, 3);
    this->args.get(&flag, 4);

    size_t bytes = input->shape_bytes();
    const OpTile &tile_out = cfg.output_tiles[0];
    size_t nelems_per_tile = tile_out.x * tile_out.y > input->shape.size()
                                 ? input->shape.size()
                                 : tile_out.x * tile_out.y;
    Dims unit_out_dims{1, 1, 1, static_cast<ark::DimType>(nelems_per_tile)};

    return Op::function_name("ark::comm::put_packet_msll",
                             {{
                                 input->ldims.dims4(),  // Dims
                                 input->shape.dims4(),  // Shape
                                 unit_out_dims,         // UnitOutDims
                                 cfg.num_warps * 32,    // NumThreads
                                 dst_rank,              // DstRank
                                 rank,                  // Rank
                                 dst_offset,            // DstOffset
                                 bytes,                 // Length
                                 flag,                  // Flag
                             }});
}

OpArgs MsllPutPacketOp::function_call_args(const OpConfig &) const {
    Tensor *input = this->inputs[0];
    Tensor *recv_buf = this->inputs[2];

    CHECK(input->buf != nullptr);
    CHECK(recv_buf->buf != nullptr);

    OpArgs opargs;
    opargs.put(static_cast<size_t>(recv_buf->buf->get_buf_offset() +
                                   recv_buf->offset_bytes()));
    opargs.put(static_cast<size_t>(input->buf->get_buf_offset() +
                                   input->offset_bytes()));
    return opargs;
}

Tensor *Model::put_packet_msll(Tensor *input, Tensor *local_tmp_buf,
                               Tensor *recv_buf, int id, int rank, int dst_rank,
                               size_t dst_offset, int flag,
                               const std::string &name) {
    CHECK(input != nullptr);
    CHECK(local_tmp_buf != nullptr);
    CHECK(recv_buf != nullptr);
    CHECK(input->is_sequential());
    if (input->ndims() > 1) {
        LOG(ERROR, "supports only 1D input");
    }

    std::string pt = "none";
    if (input->type == FP16) {
        pt = "fp16";
    }
    local_tmp_buf->exported = true;
    recv_buf->imported_rank = dst_rank;
    MsllPutPacketOp op{pt,   input,    local_tmp_buf, recv_buf, id,
                       rank, dst_rank, dst_offset,    flag,     name};
    return this->impl->add_op(op)[0];
}

MsllReduceAndWritePacketOp::MsllReduceAndWritePacketOp(
    const std::string &prec_type, std::vector<Tensor *> inputs, Tensor *output,
    int id, int rank, int npeers, size_t elems_per_rank, size_t scratch_offset,
    size_t remote_dst_offset, int flag, const std::string &name)
    : Op{OP_REDUCE_AND_WRITE_PACKET_MSLL,
         prec_type,
         inputs,
         {output},
         {{id, rank, npeers, elems_per_rank, scratch_offset, remote_dst_offset,
           flag}},
         name,
         &MsllPacketConfigMap,
         -1,
         true} {}

std::string MsllReduceAndWritePacketOp::function_name(
    const OpConfig &cfg) const {
    Tensor *input = this->inputs[0];

    int rank;
    int npeers;
    size_t elems_per_rank;
    size_t scratch_offset;
    size_t remote_dst_offset;
    int flag;

    this->args.get(&rank, 1);
    this->args.get(&npeers, 2);
    this->args.get(&elems_per_rank, 3);
    this->args.get(&scratch_offset, 4);
    this->args.get(&remote_dst_offset, 5);
    this->args.get(&flag, 6);

    Dims shape_dims = {1, 1, 1, static_cast<ark::DimType>(elems_per_rank)};
    const OpTile &tile_out = cfg.output_tiles[0];
    size_t nelems_per_tile = tile_out.x * tile_out.y > shape_dims.size()
                                 ? shape_dims.size()
                                 : tile_out.x * tile_out.y;
    Dims unit_out_dims{1, 1, 1, static_cast<ark::DimType>(nelems_per_tile)};

    return Op::function_name("ark::comm::reduce_and_write_packet_msll",
                             {{
                                 input->ldims.dims4(),  // Dims
                                 input->shape.dims4(),  // Shape
                                 unit_out_dims,         // UnitOutDims
                                 cfg.num_warps * 32,    // NumThreads
                                 npeers,                // NPeers
                                 elems_per_rank,        // NElemsPerRank
                                 rank,                  // Rank
                                 remote_dst_offset,     // RemoteDstOffset
                                 scratch_offset,        // ScratchOffset
                                 flag,                  // Flag
                             }});
}

Tensor *Model::reduce_and_write_packet_msll(
    Tensor *input, Tensor *scratch, Tensor *output,
    const std::vector<Tensor *> &remote_peer_bufs, int id, int rank, int npeers,
    size_t elems_per_rank, size_t scratch_offset, size_t remote_dst_offset,
    int flag, const std::string &name) {
    CHECK(input != nullptr);
    CHECK(output != nullptr);
    CHECK(input->is_sequential());
    CHECK(output->is_sequential());
    if (input->ndims() > 1 || output->ndims() > 1) {
        LOG(ERROR, "supports only 1D input");
    }

    std::string pt = "none";
    if (input->type == FP16) {
        pt = "fp16";
    }
    scratch->exported = true;
    for (int i = 0; i < npeers; i++) {
        int remote_rank = i < rank ? i : i + 1;
        remote_peer_bufs[i]->imported_rank = remote_rank;
    }
    std::vector<Tensor *> inputs = {input, scratch};
    inputs.insert(inputs.end(), remote_peer_bufs.begin(),
                  remote_peer_bufs.end());
    MsllReduceAndWritePacketOp op{pt,
                                  inputs,
                                  output,
                                  id,
                                  rank,
                                  npeers,
                                  elems_per_rank,
                                  scratch_offset,
                                  remote_dst_offset,
                                  flag,
                                  name};
    return this->impl->add_op(op)[0];
}

OpArgs MsllReduceAndWritePacketOp::function_call_args(const OpConfig &) const {
    Tensor *input = this->inputs[0];
    Tensor *scratch = this->inputs[1];
    Tensor *output = this->outputs[0];
    std::vector<Tensor *> peer_bufs =
        std::vector<Tensor *>(this->inputs.begin() + 2, this->inputs.end());

    CHECK(input->buf != nullptr);
    CHECK(output->buf != nullptr);

    int npeers;
    this->args.get(&npeers, 2);
    CHECK(peer_bufs.size() == (size_t)npeers);

    OpArgs opargs;
    opargs.put(output);
    opargs.put(input);
    opargs.put(scratch);
    for (int i = 0; i < get_env().num_ranks_per_host - 1; i++) {
        if (i < npeers) {
            CHECK(peer_bufs[i]->buf != nullptr);
            opargs.put((size_t)(peer_bufs[i]->buf->get_buf_offset() +
                                peer_bufs[i]->offset_bytes()));
        } else {
            opargs.put((size_t)0);
        }
    }
    return opargs;
}

MsllGetFromPacketOp::MsllGetFromPacketOp(const std::string &prec_type,
                                         Tensor *shape, Tensor *input,
                                         Tensor *output, size_t src_offset,
                                         size_t dst_offset, size_t npackets,
                                         int flag, const std::string &name)
    : Op{OP_GET_FROM_PACKET_MSLL,
         prec_type,
         {shape, input},
         {output},
         {{src_offset, dst_offset, npackets, flag}},
         name,
         &MsllPacketConfigMap,
         -1,
         true} {}

std::string MsllGetFromPacketOp::function_name(const OpConfig &cfg) const {
    Tensor *output = this->outputs[0];

    size_t src_offset;
    size_t dst_offset;
    size_t npackets;
    int flag;

    this->args.get(&src_offset, 0);
    this->args.get(&dst_offset, 1);
    this->args.get(&npackets, 2);
    this->args.get(&flag, 3);

    DimType nelems = npackets * (MSLL_PACKET_SIZE / 2 / output->type_bytes());
    Dims shape_dims = {1, 1, 1, static_cast<DimType>(nelems)};
    const OpTile &tile_out = cfg.output_tiles[0];
    size_t nelems_per_tile = tile_out.x * tile_out.y > shape_dims.size()
                                 ? shape_dims.size()
                                 : tile_out.x * tile_out.y;
    Dims unit_out_dims{1, 1, 1, static_cast<ark::DimType>(nelems_per_tile)};

    return Op::function_name("ark::comm::get_from_packet_msll",
                             {{
                                 output->ldims.dims4(),  // Dims
                                 shape_dims,             // Shape
                                 unit_out_dims,          // UnitOutDims
                                 cfg.num_warps * 32,     // NumThreads
                                 npackets,               // NPacket
                                 dst_offset,             // DstOffset
                                 src_offset,             // SrcOffset
                                 flag,                   // Flag
                             }});
}

OpArgs MsllGetFromPacketOp::function_call_args(const OpConfig &) const {
    Tensor *input = this->inputs[1];
    Tensor *output = this->outputs[0];

    CHECK(input->buf != nullptr);
    CHECK(output->buf != nullptr);

    OpArgs opargs;
    opargs.put(output);
    opargs.put(input);
    return opargs;
}

Tensor *Model::get_packet_msll(Tensor *input, Tensor *output, size_t src_offset,
                               size_t dst_offset, size_t npackets, int flag,
                               const std::string &name) {
    CHECK(input != nullptr);
    CHECK(output != nullptr);
    CHECK(input->is_sequential());
    CHECK(output->is_sequential());
    if (input->ndims() > 1 || output->ndims() > 1) {
        LOG(ERROR, "supports only 1D input");
    }

    Dims shape = {
        (DimType)(npackets * (MSLL_PACKET_SIZE / 2 / output->type_bytes()))};
    // TODO: remove this hack
    Tensor *mock_tensor = this->tensor(shape, input->type, input->buf);
    MsllGetFromPacketOp op{"none",     mock_tensor, input, output, src_offset,
                           dst_offset, npackets,    flag,  name};
    return this->impl->add_op(op)[0];
}

const OpConfigMap MsllPacketConfigMap = {
    {{OP_ARCH_CUDA_ANY, "any"},
     {// NumWarps, SmemBytes, InDepsTiles, OutDepsTiles, SyncPre, SyncPost
      {16,
       0,
       {{-1, 1024},
        {-1, -1},
        {-1, -1},
        {-1, -1},
        {-1, -1},
        {-1, -1},
        {-1, -1},
        {-1, -1},
        {-1, -1}},
       {{-1, -1}},
       false,
       false}}},
};

}  // namespace ark
