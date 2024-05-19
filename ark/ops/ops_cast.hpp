// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef ARK_OPS_CAST_HPP_
#define ARK_OPS_CAST_HPP_

#include "ops_broadcast.hpp"
#include "ops_tensor.hpp"

namespace ark {

class ModelOpCast : public ModelOpBroadcast1 {
   public:
    ModelOpCast() = default;
    ModelOpCast(ModelTensorRef input, ModelDataType data_type,
                ModelTensorRef output);
};

class ModelOpByteCast : public ModelOpTensor {
   public:
    ModelOpByteCast() = default;
    ModelOpByteCast(ModelTensorRef input, ModelDataType data_type,
                    const Dims &shape, const Dims &strides, const Dims &offsets,
                    const Dims &padded_shape);
};

}  // namespace ark

#endif  // ARK_OPS_CAST_HPP_
