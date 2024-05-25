// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef ARK_OPS_RESHAPE_HPP_
#define ARK_OPS_RESHAPE_HPP_

#include "ark/dims.hpp"
#include "ark/model.hpp"
#include "model/model_op.hpp"
#include "ops_tensor.hpp"

namespace ark {

class ModelOpReshape : public ModelOpTensor {
   public:
    ModelOpReshape() = default;
    ModelOpReshape(ModelTensorRef input, const Dims &shape, const Dims &strides,
                   const Dims &offsets);
};

}  // namespace ark

#endif  // ARK_OPS_RESHAPE_HPP_
