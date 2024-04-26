// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef ARK_OPS_TENSOR_HPP_
#define ARK_OPS_TENSOR_HPP_

#include "ark/dims.hpp"
#include "ark/model.hpp"
#include "model/model_op.hpp"

namespace ark {

class ModelOpTensor : public ModelOp {
   public:
    ModelOpTensor() = default;
    ModelOpTensor(ModelBufferRef buffer, const Dims &shape,
                  ModelDataType data_type, const Dims &strides,
                  const Dims &offsets, const Dims &pads, bool exported,
                  int imported_rank);
};

}  // namespace ark

#endif  // ARK_OPS_TENSOR_HPP_
