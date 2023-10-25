// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include <cassert>

#include "logging.h"
#include "model.h"

namespace ark {

extern const OpConfigMap MsllGatherFromPeersConfigMap;
constexpr int MAX_PEER_NUM = 7;

// currently only support in single node
MsllGatherFromPeersOp::MsllGatherFromPeersOp(
    const std::string &prec_type, Tensor *local_buf, Tensor *trans_region_local,
    std::vector<Tensor *> remote_bufs, Tensor *trans_region_remote, int sid,
    int rank, int npeers, size_t chunk_bytes, const std::string &name)
    : Op(OP_GATHER_FROM_PEERS_MSLL, prec_type, {trans_region_remote},
         {trans_region_local, local_buf}, {{rank, npeers, sid, chunk_bytes}},
         name, &MsllGatherFromPeersConfigMap, -1, true) {
    this->inputs.insert(this->inputs.end(), remote_bufs.begin(),
                        remote_bufs.end());
}

std::string MsllGatherFromPeersOp::function_name(const OpConfig &cfg) const {
    Tensor *dst_buff = this->outputs[0];
    CHECK(dst_buff->is_sequential());

    int rank;
    int npeers;
    size_t chunk_bytes;
    this->args.get(&rank, 0);
    this->args.get(&npeers, 1);
    this->args.get(&chunk_bytes, 3);

    const OpTile &tile_out = cfg.output_tiles[0];
    size_t ntitle_eles = tile_out.x * tile_out.y > dst_buff->shape.size()
                             ? dst_buff->shape.size()
                             : tile_out.x * tile_out.y;
    Dims unit_out_dims{1, 1, 1, static_cast<ark::DimType>(ntitle_eles)};
    Dims shape_dims = {1, 1, 1, (int64_t)chunk_bytes / dst_buff->type_bytes()};
    Dims dims = dst_buff->ldims.dims4();

    return Op::function_name("ark::comm::gather_from_peers_msll",
                             {{dims,                // Dims
                               shape_dims,          // Shape
                               unit_out_dims,       // UnitOutDims
                               cfg.num_warps * 32,  // NumThreads
                               npeers, rank, chunk_bytes}});
}

OpArgs MsllGatherFromPeersOp::function_call_args(const OpConfig &) const {
    int rank;
    int npeers;
    this->args.get(&rank, 0);
    this->args.get(&npeers, 1);

    Tensor *local_buff = this->outputs[1];
    std::vector<Tensor *> remote_bufs =
        std::vector<Tensor *>(this->inputs.begin() + 1, this->inputs.end());

    CHECK(local_buff->buf != nullptr);

    OpArgs opargs;
    // gether_from_peers_msll(dst_offset, src_offset...)
    opargs.put((size_t)(local_buff->buf->get_buf_offset() +
                        local_buff->offset_bytes()));
    for (int i = 0; i < MAX_PEER_NUM; i++) {
        if (i < npeers) {
            CHECK(remote_bufs[i]->buf != nullptr);
            opargs.put((size_t)(remote_bufs[i]->buf->get_buf_offset() +
                                remote_bufs[i]->offset_bytes()));
        } else {
            opargs.put((size_t)0);
        }
    }
    opargs.put(remote_bufs[0]);
    return opargs;
}

Tensor *Model::gather_from_peers_msll(Tensor *input, int sid, int npeers,
                                      size_t chunk_bytes,
                                      const std::string &name) {
    LOG(DEBUG, "gather_from_peers_msll ", input->shape, " npeers ", npeers);
    input->exported = true;

    int rank = this->impl->rank;
    std::vector<Tensor *> remote_bufs;
    for (int i = 0; i < npeers; i++) {
        int peer_rank = i < rank ? i : i + 1;
        Tensor *remote_buf = this->tensor(input->shape, input->type);
        remote_buf->imported_rank = peer_rank;
        remote_bufs.push_back(remote_buf);
    }
    std::string pt = "none";
    if (input->type == FP16) {
        pt = "fp16";
    }
    Dims shape = {(long long)chunk_bytes / input->type_bytes()};
    // These two tensors are not actually used, just give hint to scheduler to
    // split to correct tiles
    Tensor *trans_region_local = this->tensor(shape, input->type, input->buf);
    Tensor *trans_region_remote = this->tensor(shape, input->type, input->buf);
    MsllGatherFromPeersOp op{pt,
                             input,
                             trans_region_local,
                             remote_bufs,
                             trans_region_remote,
                             sid,
                             this->impl->rank,
                             npeers,
                             chunk_bytes,
                             name};
    return this->impl->add_op(op)[1];
}

Tensor *Model::local_all_gather_msll(Tensor *input, int gpu_id,
                                     int ngpus_per_node,
                                     const std::string &name) {
    assert(input != nullptr);
    if (!input->is_sequential()) {
        LOG(WARN,
            "all_gather may not work correctly if the input tensor is "
            "not contiguous");
    }
    ark::Dims ori_shape = input->shape;
    Tensor *input_reshaped = this->reshape(input, {input->shape.size()});
    int npeers = ngpus_per_node - 1;
    int id = this->impl->next_eid;
    LOG(DEBUG, "local_all_gather_msll ", input->shape, " ", gpu_id, " ", id,
        " ", ngpus_per_node, " ", ngpus_per_node, " ");
    Tensor *tensor = this->device_sync_msll(input, ngpus_per_node);
    // seems we can change the offset of input for the input based on gpu id
    assert(tensor->shape.size() % ngpus_per_node == 0);
    size_t bytes_per_chunk = tensor->shape_bytes() / ngpus_per_node;
    Tensor *out =
        this->gather_from_peers_msll(tensor, id, npeers, bytes_per_chunk, name);
    this->impl->next_eid += 1;
    return this->reshape(out, ori_shape);
}

const OpConfigMap MsllGatherFromPeersConfigMap = {
    {{OP_ARCH_CUDA_ANY, "any"},
     {// NumWarps, SmemBytes, InDepsTiles, OutDepsTiles, SyncPre, SyncPost
      // TODO: The config for 32MB elements, need to update for other message
      // size
      {16,
       0,
       {{-1, 65536},
        {-1, -1},
        {-1, -1},
        {-1, -1},
        {-1, -1},
        {-1, -1},
        {-1, -1},
        {-1, -1}},
       {{1, 65536}, {-1, -1}},
       false,
       true}}},
};
}  // namespace ark
