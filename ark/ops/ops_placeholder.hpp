// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef ARK_OPS_PLACEHOLDER_HPP_
#define ARK_OPS_PLACEHOLDER_HPP_

#include "ark/model.hpp"
#include "model/model_op.hpp"

namespace ark {

class ModelOpPlaceholder : public ModelOp {
   public:
    ModelOpPlaceholder() = default;
    ModelOpPlaceholder(ModelBufferRef buffer, const Dims &shape,
                       ModelDataType data_type, const Dims &strides,
                       const Dims &offsets, const Dims &padded_shape,
                       void *data = nullptr);
};

}  // namespace ark

#endif  // ARK_OPS_PLACEHOLDER_HPP_