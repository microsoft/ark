// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "ops_rope.hpp"

#include "ops_common.hpp"

namespace ark {

ModelOpRope::ModelOpRope(ModelTensorRef input, ModelTensorRef other,
                         ModelTensorRef output)
    : ModelOpBroadcast2("Rope", input, other, output) {}

Tensor Model::rope(Tensor input, Tensor other, Tensor output,
                   const std::string &name) {
    return impl_
        ->create_op<ModelOpRope>(name, input.ref_, other.ref_, output.ref_)
        ->result_tensors()[0];
}

}  // namespace ark
