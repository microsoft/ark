// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef ARK_OPS_SCALE_HPP_
#define ARK_OPS_SCALE_HPP_

#include "ark/dims.hpp"
#include "ark/model.hpp"
#include "model/model_op.hpp"

namespace ark {

class ModelOpScale : public ModelOp {
   public:
    ModelOpScale() = default;
    ModelOpScale(ModelTensorRef input, float val, ModelTensorRef output);

    std::string impl_name(const nlohmann::json &config) const override;

    std::vector<ModelOpArg> impl_args(
        [[maybe_unused]] const nlohmann::json &config) const override;

    nlohmann::ordered_json default_config() const override;
};

}  // namespace ark

#endif  // ARK_OPS_SCALE_HPP_
