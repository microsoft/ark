// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef ARK_OPS_SCALAR_HPP_
#define ARK_OPS_SCALAR_HPP_

#include "ark/data_type.hpp"
#include "ops_broadcast.hpp"

namespace ark {

class ModelOpScalarAssign : public ModelOp {
   public:
    ModelOpScalarAssign() = default;
    ModelOpScalarAssign(float val, const Dims &shape, ModelDataType data_type,
                        ModelTensorRef output);

    std::string impl_name(const Json &config) const override;

    std::vector<ModelOpArg> impl_args([
        [maybe_unused]] const Json &config) const override;

    Json default_config() const override;
};

class ModelOpScalarAdd : public ModelOpBroadcast1 {
   public:
    ModelOpScalarAdd() = default;
    ModelOpScalarAdd(ModelTensorRef input, float val, ModelTensorRef output);

    std::vector<ModelOpArg> impl_args([
        [maybe_unused]] const Json &config) const override;
};

class ModelOpScalarMul : public ModelOpBroadcast1 {
   public:
    ModelOpScalarMul() = default;
    ModelOpScalarMul(ModelTensorRef input, float val, ModelTensorRef output);

    std::vector<ModelOpArg> impl_args([
        [maybe_unused]] const Json &config) const override;
};

}  // namespace ark

#endif  // ARK_OPS_SCALAR_HPP_
