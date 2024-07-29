// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "ops_copy.hpp"

#include "ops_common.hpp"

namespace ark {

ModelOpCopy::ModelOpCopy(ModelTensorRef input, ModelTensorRef output)
    : ModelOpBroadcast1(
          "Copy", input,
          output ? output
                 : std::make_shared<ModelTensor>(
                       input->data_type(), std::make_shared<ModelBuffer>(),
                       input->shape())) {
    if (output) {
        check_match_data_type(input, output);
    }
    verify();
}

Tensor Model::copy(Tensor input, Tensor output, const std::string &config,
                   const std::string &name) {
    return impl_->create_op<ModelOpCopy>(config, name, input.ref_, output.ref_)
        ->result_tensors()[0];
}

}  // namespace ark
