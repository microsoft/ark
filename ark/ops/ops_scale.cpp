// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "ops_scale.hpp"

#include "ops_common.hpp"

namespace ark {

ModelOpScale::ModelOpScale(ModelTensorRef input, float factor,
                           ModelTensorRef output)
    : ModelOpBroadcast1(
          "Scale", input,
          output ? output
                 : std::make_shared<ModelTensor>(
                       input->data_type(), std::make_shared<ModelBuffer>(),
                       input->shape())) {
    if (output) {
        check_match_data_type(input, output);
    }
    args_ = {{"Factor", factor}};

    verify();
}

std::vector<ModelOpArg> ModelOpScale::impl_args(
    [[maybe_unused]] const nlohmann::json &config) const {
    float factor = args_.at("Factor").value<float>();
    return {result_tensors_[0], read_tensors_[0], factor};
}

Tensor Model::scale(Tensor input, float factor, Tensor output,
                    const std::string &name) {
    return impl_->create_op<ModelOpScale>(name, input.ref_, factor, output.ref_)
        ->result_tensors()[0];
}

}  // namespace ark
