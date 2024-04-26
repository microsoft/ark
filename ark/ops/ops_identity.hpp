// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef ARK_OPS_IDENTITY_HPP_
#define ARK_OPS_IDENTITY_HPP_

#include "ark/dims.hpp"
#include "ark/model.hpp"
#include "model/model_op.hpp"
#include "ops_tensor.hpp"

namespace ark {

class ModelOpIdentity : public ModelOpTensor {
   public:
    ModelOpIdentity() = default;
    ModelOpIdentity(ModelTensorRef input,
                    const std::vector<ModelTensorRef> &deps);
};

}  // namespace ark

#endif  // ARK_OPS_IDENTITY_HPP_
