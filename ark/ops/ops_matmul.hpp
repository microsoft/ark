// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef ARK_OPS_MATMUL_HPP_
#define ARK_OPS_MATMUL_HPP_

#include "ark/dims.hpp"
#include "ark/model.hpp"
#include "model/model_op.hpp"

namespace ark {

class ModelOpMatmul : public ModelOp {
   public:
    ModelOpMatmul() = default;
    ModelOpMatmul(ModelTensorRef input, ModelTensorRef other,
                  ModelTensorRef output, bool trans_input, bool trans_other);
};

}  // namespace ark

#endif  // ARK_OPS_MATMUL_HPP_
