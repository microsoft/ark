// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef ARK_OPS_NOOP_HPP_
#define ARK_OPS_NOOP_HPP_

#include "model/model_op.hpp"

namespace ark {

class ModelOpNoop : public ModelOp {
   public:
    ModelOpNoop() = default;
    ModelOpNoop(ModelTensorRef input);

    std::string impl_name(const nlohmann::json &config) const override;

    std::vector<ModelOpArg> impl_args(
        [[maybe_unused]] const nlohmann::json &config) const override;

    nlohmann::ordered_json default_config() const override;
};

}  // namespace ark

#endif  // ARK_OPS_NOOP_HPP_
