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
                                               const std::string &name)
    : Op(OP_READ_AND_REDUCE_MSCCLPP, prec_type, {local_buf, remote_buf},
         {local_buf}, {{rank, src_rank, sid}}, name,
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
    this->args.get(&rank, 0);
    this->args.get(&peer_rank, 1);

    const OpTile &tile_out = cfg.output_tiles[0];
    Dims unit_out_dims{1, 1, tile_out.x, tile_out.y};

    return Op::function_name("ark::comm::read_and_reduce_mscclpp",
                             {{src_buf->ldims.dims4(),  // InDims
                               src_buf->shape.dims4(),  // InShape
                               dst_buff->ldims.dims4(), // OutDims
                               dst_buff->shape.dims4(), // OutShape
                               unit_out_dims,           // UnitOutDims
                               cfg.num_warps * 32,      // NumThreads
                               peer_rank,
                               rank}});
}

OpArgs MscclppReadAndReduceOp::function_call_args(const OpConfig &) const
{
    Tensor *input = this->inputs[0];
    Tensor *recvbuf = this->inputs[1];

    CHECK(input->buf != nullptr);
    CHECK(recvbuf->buf != nullptr);

    OpArgs opargs;
    // read_and_redcue_mscclpp(dst_offset, src_offset...)
    opargs.put((int)(input->buf->get_buf_offset() + input->offset_bytes()));
    opargs.put((int)(recvbuf->buf->get_buf_offset() + recvbuf->offset_bytes()));
    return opargs;
}

Tensor *Model::read_and_reduce_mscclpp(Tensor *input, int sid, int src_rank,
                                       const std::string &name)
{
    LOG(DEBUG, "read_and_reduce_mscclpp ", input->shape, " ", src_rank);
    input->exported = true;

    Tensor *remote_buf = this->tensor(input->shape, input->type);
    remote_buf->imported_rank = src_rank;
    MscclppReadAndReduceOp op{OP_PREC_NONE,     input,    remote_buf, sid,
                              this->impl->rank, src_rank, name};
    return this->impl->add_op(op)[0];
}

Tensor *Model::local_reduce_scatter_mscclpp(Tensor *input, int gpu_id, int begin_sid,
                                            int ngpus_per_node,
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
    // Assume the input tensor is 1D
    int axis = 0;
    assert(input->shape[axis]  % ngpus_per_node == 0);
    DimType ndim_per_shape = input->shape[axis] / ngpus_per_node;
    LOG(DEBUG, "local_reduce_scatter_mscclpp ", input->shape, " ", gpu_id, " ",
              begin_sid, " ", ngpus_per_node, " ", npeers, " ", axis, " ", ndim_per_shape);
    Tensor *out = this->device_sync_mscclpp(ngpus_per_node);
    std::vector<Tensor *> tensors =
        this->sharding(this->identity(input, {out}), axis, ndim_per_shape,
                       "reduce_scatter_sharding");
    // seems we can change the offset of input for the input based on gpu id
    for (int i = 0; i < npeers; ++i) {
        int peer_rank = i < gpu_id ? i : i + 1;
        this->read_and_reduce_mscclpp(tensors[peer_rank], begin_sid + peer_rank,
                                      peer_rank, name);
    }
    return input;
}

const OpConfigMap MscclppReadAndReduceConfigMap = {
    {{OP_ARCH_CUDA_ANY, OP_PREC_NONE},
     {
         // NumWarps, SmemBytes, InDepsTiles, OutDepsTiles, SyncPre, SyncPost
         {16, 0, {{-1, -1}, {-1, -1}}, {{-1, -1}}, false, true},
     }},
};
}; // namespace ark
