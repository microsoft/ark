// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef ARK_OPS_REFER_HPP_
#define ARK_OPS_REFER_HPP_

#include "ops_tensor.hpp"

namespace ark {

class ModelOpRefer : public ModelOpTensor {
   public:
    ModelOpRefer() = default;
    ModelOpRefer(ModelTensorRef input, const Dims &shape, const Dims &strides,
                 const Dims &offsets, const Dims &pads);
};

}  // namespace ark

#endif  // ARK_OPS_REFER_HPP_
