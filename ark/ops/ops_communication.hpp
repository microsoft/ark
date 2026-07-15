// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef ARK_OPS_COMMUNICATION_HPP_
#define ARK_OPS_COMMUNICATION_HPP_

#include "model/model_op.hpp"

namespace ark {

class ModelOpSend : public ModelOp {
   public:
    ModelOpSend() = default;
    ModelOpSend(ModelTensorRef input, int remote_rank, int tag,
                ModelTensorRef output);

    std::string impl_name(const Json &config) const override;

    std::vector<ModelOpArg> impl_args(const Json &config) const override;

    Json default_config(const ArchRef arch = ARCH_ANY) const override;
};

class ModelOpSendDone : public ModelOp {
   public:
    ModelOpSendDone() = default;
    ModelOpSendDone(ModelTensorRef input);

    std::string impl_name(const Json &config) const override;

    std::vector<ModelOpArg> impl_args(const Json &config) const override;

    Json default_config(const ArchRef arch = ARCH_ANY) const override;
};

class ModelOpRecv : public ModelOp {
   public:
    ModelOpRecv() = default;
    ModelOpRecv(ModelTensorRef output, int remote_rank, int tag);

    std::string impl_name(const Json &config) const override;

    std::vector<ModelOpArg> impl_args(const Json &config) const override;

    Json default_config(const ArchRef arch = ARCH_ANY) const override;
};

class ModelOpSendPacket : public ModelOp {
   public:
    ModelOpSendPacket() = default;
    ModelOpSendPacket(ModelTensorRef input, int remote_rank, int tag,
                      uint32_t flag, ModelTensorRef output);

    std::string impl_name(const Json &config) const override;

    std::vector<ModelOpArg> impl_args(const Json &config) const override;

    Json default_config(const ArchRef arch = ARCH_ANY) const override;
};

class ModelOpRecvPacket : public ModelOp {
   public:
    ModelOpRecvPacket() = default;
    ModelOpRecvPacket(ModelTensorRef output, int remote_rank, int tag,
                      uint32_t flag, ModelTensorRef scratch);

    std::string impl_name(const Json &config) const override;

    std::vector<ModelOpArg> impl_args(const Json &config) const override;

    Json default_config(const ArchRef arch = ARCH_ANY) const override;
};

class ModelOpRecvReduceSendPacket : public ModelOp {
   public:
    ModelOpRecvReduceSendPacket() = default;
    ModelOpRecvReduceSendPacket(ModelTensorRef input, ModelTensorRef output,
                                int rank, const std::vector<int> &remote_rank,
                                int recv_tag, int output_tag, uint32_t flag,
                                std::vector<ModelTensorRef> &peer_output_refs,
                                ModelTensorRef scratch);

    std::string impl_name(const Json &config) const override;

    std::vector<ModelOpArg> impl_args(const Json &config) const override;

    Json default_config(const ArchRef arch = ARCH_ANY) const override;
};

class ModelOpRecvReduceSend : public ModelOp {
   public:
    ModelOpRecvReduceSend() = default;
    ModelOpRecvReduceSend(ModelTensorRef input, ModelTensorRef output, int rank,
                          const std::vector<int> &remote_rank, int recv_tag,
                          int output_tag,
                          std::vector<ModelTensorRef> &peer_output_refs,
                          ModelTensorRef scratch);

    std::string impl_name(const Json &config) const override;

    std::vector<ModelOpArg> impl_args(const Json &config) const override;

    Json default_config(const ArchRef arch = ARCH_ANY) const override;
};

// Tile-local one-shot packet allreduce (the only packet all-reduce op). Block
// `uop_idx` reduces ONLY its own [TileRows, TileCols] column-tile of the
// [Rows, Cols] input — the tile a fused preceding matmul wrote on the same
// block — so the matmul→AR handoff is intra-block (no device-wide barrier) and
// each tile's exchange overlaps other tiles' matmul. A tile grid covering the
// whole tensor gives the plain one-shot behavior. The tile is chosen by the
// planner (config "Tile"), like other ops. See allreduce_packet in comm.h.
class ModelOpAllReducePacket : public ModelOp {
   public:
    ModelOpAllReducePacket() = default;
    ModelOpAllReducePacket(
        ModelTensorRef input, ModelTensorRef output, int rank, int rank_num,
        const std::vector<ModelTensorRef> &peer_output_refs);

    std::string impl_name(const Json &config) const override;

    std::vector<ModelOpArg> impl_args(const Json &config) const override;

    Json default_config(const ArchRef arch = ARCH_ANY) const override;
};

class ModelOpDeviceSync : public ModelOp {
   public:
    ModelOpDeviceSync() = default;
    ModelOpDeviceSync(ModelTensorRef input, int rank, int rank_num,
                      ModelTensorRef output);

    std::string impl_name(const Json &config) const override;

    std::vector<ModelOpArg> impl_args(const Json &config) const override;

    Json default_config(const ArchRef arch = ARCH_ANY) const override;
};

// Read-based Reduce-Scatter + All-Gather all-reduce for LARGE messages
// (bandwidth-optimal, O(N) traffic). Grid-wide: every block runs all three
// phases (scatter / reduce-scatter / all-gather) with grid + cross-rank
// barriers. Standalone op (its own processor group), not tile-local. See
// allreduce_rsag in comm.h.
class ModelOpAllReduceRsag : public ModelOp {
   public:
    ModelOpAllReduceRsag() = default;
    ModelOpAllReduceRsag(ModelTensorRef input, ModelTensorRef output, int rank,
                         int rank_num,
                         const std::vector<ModelTensorRef> &peer_output_refs);

    std::string impl_name(const Json &config) const override;

    std::vector<ModelOpArg> impl_args(const Json &config) const override;

    Json default_config(const ArchRef arch = ARCH_ANY) const override;
};

// All-pairs one-shot LL-packet all-reduce for SMALL messages, a port of
// mscclpp's `allreduceAllPairs` (the NCCL-API <=16KB path). Standalone
// grid-strided op (its own processor group; NOT tile-local): every block owns a
// grid-strided stripe of the whole input, writes it to all peers as LL packets,
// then reduces its stripe. See allreduce_allpair_packet in comm.h.
class ModelOpAllReduceAllpairPacket : public ModelOp {
   public:
    ModelOpAllReduceAllpairPacket() = default;
    ModelOpAllReduceAllpairPacket(
        ModelTensorRef input, ModelTensorRef output, int rank, int rank_num,
        const std::vector<ModelTensorRef> &peer_output_refs);

    std::string impl_name(const Json &config) const override;

    std::vector<ModelOpArg> impl_args(const Json &config) const override;

    Json default_config(const ArchRef arch = ARCH_ANY) const override;
};
}  // namespace ark

#endif  // ARK_OPS_COMMUNICATION_HPP_
