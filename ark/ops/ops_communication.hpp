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

    std::vector<ModelOpArg> impl_args([
        [maybe_unused]] const Json &config) const override;

    Json default_config(const Arch &arch = ARCH_ANY) const override;
};

class ModelOpSendDone : public ModelOp {
   public:
    ModelOpSendDone() = default;
    ModelOpSendDone(ModelTensorRef input);

    std::string impl_name(const Json &config) const override;

    std::vector<ModelOpArg> impl_args([
        [maybe_unused]] const Json &config) const override;

    Json default_config(const Arch &arch = ARCH_ANY) const override;
};

class ModelOpRecv : public ModelOp {
   public:
    ModelOpRecv() = default;
    ModelOpRecv(ModelTensorRef output, int remote_rank, int tag);

    std::string impl_name(const Json &config) const override;

    std::vector<ModelOpArg> impl_args([
        [maybe_unused]] const Json &config) const override;

    Json default_config(const Arch &arch = ARCH_ANY) const override;
};

}  // namespace ark

#endif  // ARK_OPS_COMMUNICATION_HPP_
