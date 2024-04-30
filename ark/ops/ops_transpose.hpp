// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef ARK_OPS_TRANSPOSE_HPP_
#define ARK_OPS_TRANSPOSE_HPP_

#include "ark/dims.hpp"
#include "ark/model.hpp"
#include "model/model_op.hpp"

namespace ark {

class ModelOpTranspose : public ModelOp {
   public:
    ModelOpTranspose() = default;
    ModelOpTranspose(ModelTensorRef input, const Dims &permutation,
                     ModelTensorRef output);

    std::string impl_name(const nlohmann::json &config) const override;

    std::vector<ModelOpArg> impl_args(
        [[maybe_unused]] const nlohmann::json &config) const override;

    nlohmann::ordered_json default_config() const override;
};

}  // namespace ark

#endif  // ARK_OPS_TRANSPOSE_HPP_
