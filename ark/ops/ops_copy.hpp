// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef ARK_OPS_COPY_HPP_
#define ARK_OPS_COPY_HPP_

#include "ops_broadcast.hpp"

namespace ark {

class ModelOpCopy : public ModelOpBroadcast1 {
   public:
    ModelOpCopy() = default;
    ModelOpCopy(ModelTensorRef input, ModelTensorRef output);
};

}  // namespace ark

#endif  // ARK_OPS_COPY_HPP_
