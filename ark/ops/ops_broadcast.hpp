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

    std::string impl_name(const nlohmann::json &config) const override;

    std::vector<ModelOpArg> impl_args(
        [[maybe_unused]] const nlohmann::json &config) const override;

    nlohmann::ordered_json default_config() const override;
};

}  // namespace ark

#endif  // ARK_OPS_BROADCAST_HPP_
