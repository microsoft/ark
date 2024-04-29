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

ModelTensorRef Model::copy(ModelTensorRef input, ModelTensorRef output,
                           const std::string &name) {
    return impl_->create_op<ModelOpCopy>(name, input, output)
        ->result_tensors()[0];
}

}  // namespace ark
