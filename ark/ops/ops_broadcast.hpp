// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef ARK_OPS_BROADCAST_HPP_
#define ARK_OPS_BROADCAST_HPP_

#include "model/model_op.hpp"

namespace ark {

class ModelOpBroadcast1 : public ModelOp {
   public:
    ModelOpBroadcast1() = default;
    ModelOpBroadcast1(const std::string &type_name, ModelTensorRef input,
                      ModelTensorRef output);

    std::string impl_name(const Json &config) const override;

    std::vector<ModelOpArg> impl_args(const Json &config) const override;

    Json default_config(const ArchRef arch = ARCH_ANY) const override;
};

class ModelOpBroadcast2 : public ModelOp {
   public:
    ModelOpBroadcast2() = default;
    ModelOpBroadcast2(const std::string &type_name, ModelTensorRef input,
                      ModelTensorRef other, ModelTensorRef output);

    std::string impl_name(const Json &config) const override;

    std::vector<ModelOpArg> impl_args(const Json &config) const override;

    Json default_config(const ArchRef arch = ARCH_ANY) const override;
};

}  // namespace ark

#endif  // ARK_OPS_BROADCAST_HPP_
