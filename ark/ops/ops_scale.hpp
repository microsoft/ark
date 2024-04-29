// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef ARK_OPS_SCALE_HPP_
#define ARK_OPS_SCALE_HPP_

#include "ops_broadcast.hpp"

namespace ark {

class ModelOpScale : public ModelOpBroadcast1 {
   public:
    ModelOpScale() = default;
    ModelOpScale(ModelTensorRef input, float val, ModelTensorRef output);

    std::vector<ModelOpArg> impl_args(
        [[maybe_unused]] const nlohmann::json &config) const override;
};

}  // namespace ark

#endif  // ARK_OPS_SCALE_HPP_
