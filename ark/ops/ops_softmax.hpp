// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef ARK_OPS_SOFTMAX_HPP_
#define ARK_OPS_SOFTMAX_HPP_

#include "model/model_op.hpp"

namespace ark {

class ModelOpSoftmax : public ModelOp {
   public:
    ModelOpSoftmax() = default;
    ModelOpSoftmax(ModelTensorRef input, ModelTensorRef output);

    std::string impl_name(const Json &config) const override;

    std::vector<ModelOpArg> impl_args(const Json &config) const override;

    Json default_config(const ArchRef arch = ARCH_ANY) const override;
};

}  // namespace ark

#endif  // ARK_OPS_SOFTMAX_HPP_
