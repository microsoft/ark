// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "logging.h"
#include "model.h"
#include <cassert>

namespace ark {

extern const OpConfigMap MscclppReadAndReduceConfigMap;

// currently only support in single node
MscclppReadAndReduceOp::MscclppReadAndReduceOp(OpPrecType prec_type,
                                               Tensor *local_buf,
                                               Tensor *remote_buf, int sid,
                                               int rank, int src_rank,
                                               size_t offset, size_t bytes,
                                               const std::string &name)
    : Op(OP_READ_AND_REDUCE_MSCCLPP, prec_type, {local_buf, remote_buf},
         {local_buf}, {{rank, src_rank, sid, offset, bytes}}, name,
         &MscclppReadAndReduceConfigMap, -1, true)
{
}

std::string MscclppReadAndReduceOp::function_name(const OpConfig &cfg) const
{
    Tensor *dst_buff = this->inputs[0];
    Tensor *src_buf = this->inputs[1];
    CHECK(dst_buff->is_sequential());

    int rank;
    int peer_rank;
    size_t offset;
    size_t bytes;
    this->args.get(&rank, 0);
    this->args.get(&peer_rank, 1);
    this->args.get(&offset, 3);
    this->args.get(&bytes, 4);

    const OpTile &tile_out = cfg.output_tiles[0];
    Dims unit_out_dims{1, 1, 1, static_cast<long long>(bytes) / src_buf->type_bytes()};
    Dims shape_dims = {1, 1, 1, static_cast<long long>(bytes) / src_buf->type_bytes()};
    Dims dims = src_buf->ldims.dims4();

    return Op::function_name("ark::comm::read_and_reduce_mscclpp",
                             {{dims,               // Dims
                               shape_dims,         // Shape
                               unit_out_dims,      // UnitOutDims
                               cfg.num_warps * 32, // NumThreads
                               peer_rank, rank, offset}});
}

OpArgs MscclppReadAndReduceOp::function_call_args(const OpConfig &) const
{
    Tensor *local_buff = this->inputs[0];
    Tensor *remote_buff = this->inputs[1];

    CHECK(local_buff->buf != nullptr);
    CHECK(remote_buff->buf != nullptr);

    int rank;
    int peer_rank;
    this->args.get(&rank, 0);
    this->args.get(&peer_rank, 1);

    OpArgs opargs;
    // read_and_redcue_mscclpp(dst_offset, src_offset...)
    opargs.put(
        (int)(local_buff->buf->get_buf_offset() + local_buff->offset_bytes()));
    opargs.put((int)(remote_buff->buf->get_buf_offset() +
                     remote_buff->offset_bytes()));
    return opargs;
}

Tensor *Model::read_and_reduce_mscclpp(Tensor *input, int sid, int src_rank,
                                       size_t offset, size_t bytes,
                                       const std::string &name)
{
    LOG(DEBUG, "read_and_reduce_mscclpp ", input->shape, " ", src_rank);
    input->exported = true;

    Tensor *remote_buf = this->tensor(input->shape, input->type);
    remote_buf->imported_rank = src_rank;
    MscclppReadAndReduceOp op{
        OP_PREC_NONE, input,  remote_buf, sid, this->impl->rank,
        src_rank,     offset, bytes,      name};
    return this->impl->add_op(op)[0];
}

Tensor *Model::local_reduce_scatter_mscclpp(Tensor *input, int gpu_id,
                                            int sid, int ngpus_per_node,
                                            const std::string &name)
{
    assert(input != nullptr);
    if (input->ndims() > 1) {
        LOG(ERROR, "supports only 1D input");
    }
    if (!input->is_sequential()) {
        LOG(WARN, "reduce_scatter may not work correctly if the input tensor is "
                  "not contiguous");
    }
    int npeers = ngpus_per_node - 1;
    LOG(DEBUG, "local_reduce_scatter_mscclpp ", input->shape, " ", gpu_id, " ",
        sid, " ", ngpus_per_node, " ", npeers, " ");
    Tensor *out = this->device_sync_mscclpp(ngpus_per_node);
    Tensor * tensor = this->identity(input, {out});
    // seems we can change the offset of input for the input based on gpu id
    assert(tensor->shape.size() % ngpus_per_node == 0);
    size_t bytes_per_peer = tensor->shape_bytes() / ngpus_per_node;
    for (int i = 0; i < npeers; ++i) {
        int peer_rank = i < gpu_id ? i : i + 1;
        this->read_and_reduce_mscclpp(tensor, sid, peer_rank,
                                      bytes_per_peer * gpu_id, bytes_per_peer,
                                      name);
    }
    return input;
}

const OpConfigMap MscclppReadAndReduceConfigMap = {
    {{OP_ARCH_CUDA_ANY, OP_PREC_NONE},
     {
         // NumWarps, SmemBytes, InDepsTiles, OutDepsTiles, SyncPre, SyncPost
         {16, 0, {{-1, 1024}, {-1, 1024}}, {{-1, 1024}}, false, true},
     }},
};
}; // namespace ark
