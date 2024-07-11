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


class ModelOpDeviceSync : public ModelOp {
   public:
    ModelOpDeviceSync() = default;
    ModelOpDeviceSync(ModelTensorRef input, int rank, int rank_num,
                      ModelTensorRef output);

    std::string impl_name(const Json &config) const override;

    std::vector<ModelOpArg> impl_args(const Json &config) const override;

    Json default_config(const ArchRef arch = ARCH_ANY) const override;
};
}  // namespace ark

#endif  // ARK_OPS_COMMUNICATION_HPP_
