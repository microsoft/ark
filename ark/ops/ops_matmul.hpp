// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef ARK_OPS_MATMUL_HPP_
#define ARK_OPS_MATMUL_HPP_

#include "model/model_op.hpp"

namespace ark {

class ModelOpMatmul : public ModelOp {
   public:
    ModelOpMatmul() = default;
    ModelOpMatmul(ModelTensorRef input, ModelTensorRef other,
                  ModelTensorRef output, bool trans_input, bool trans_other);

    std::string impl_name(const json &config) const override;

    std::vector<ModelOpArg> impl_args([
        [maybe_unused]] const json &config) const override;

    nlohmann::ordered_json default_config() const override;
};

}  // namespace ark

#endif  // ARK_OPS_MATMUL_HPP_
