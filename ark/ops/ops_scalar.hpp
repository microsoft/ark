// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef ARK_OPS_SCALAR_HPP_
#define ARK_OPS_SCALAR_HPP_

#include "ops_broadcast.hpp"

namespace ark {

class ModelOpScalarAdd : public ModelOpBroadcast1 {
   public:
    ModelOpScalarAdd() = default;
    ModelOpScalarAdd(ModelTensorRef input, float val, ModelTensorRef output);

    std::vector<ModelOpArg> impl_args([
        [maybe_unused]] const nlohmann::json &config) const override;
};

class ModelOpScalarMul : public ModelOpBroadcast1 {
   public:
    ModelOpScalarMul() = default;
    ModelOpScalarMul(ModelTensorRef input, float val, ModelTensorRef output);

    std::vector<ModelOpArg> impl_args([
        [maybe_unused]] const nlohmann::json &config) const override;
};

}  // namespace ark

#endif  // ARK_OPS_SCALAR_HPP_
