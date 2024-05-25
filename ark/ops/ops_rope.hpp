// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef ARK_OPS_ROPE_HPP_
#define ARK_OPS_ROPE_HPP_

#include "ops_broadcast.hpp"

namespace ark {

class ModelOpRope : public ModelOpBroadcast2 {
   public:
    ModelOpRope() = default;
    ModelOpRope(ModelTensorRef input, ModelTensorRef weight,
                ModelTensorRef output);
};

}  // namespace ark

#endif  // ARK_OPS_ROPE_HPP_
