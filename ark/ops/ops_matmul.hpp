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

    std::string impl_name(const Json &config) const override;

    std::vector<ModelOpArg> impl_args(const Json &config) const override;

    Json default_config(const Arch &arch = ARCH_ANY) const override;
};

}  // namespace ark

#endif  // ARK_OPS_MATMUL_HPP_
