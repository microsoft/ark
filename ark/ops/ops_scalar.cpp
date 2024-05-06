// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "ops_scalar.hpp"

#include "ops_common.hpp"

namespace ark {

ModelOpScalarAdd::ModelOpScalarAdd(ModelTensorRef input, float factor,
                                   ModelTensorRef output)
    : ModelOpBroadcast1(
          "ScalarAdd", input,
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

std::vector<ModelOpArg> ModelOpScalarAdd::impl_args([
    [maybe_unused]] const nlohmann::json &config) const {
    float factor = args_.at("Factor").value<float>();
    return {result_tensors_[0], read_tensors_[0], factor};
}

ModelOpScalarMul::ModelOpScalarMul(ModelTensorRef input, float factor,
                                   ModelTensorRef output)
    : ModelOpBroadcast1(
          "ScalarMul", input,
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

std::vector<ModelOpArg> ModelOpScalarMul::impl_args([
    [maybe_unused]] const nlohmann::json &config) const {
    float factor = args_.at("Factor").value<float>();
    return {result_tensors_[0], read_tensors_[0], factor};
}

Tensor Model::add(Tensor input, float value, Tensor output,
                  const std::string &name) {
    return impl_
        ->create_op<ModelOpScalarAdd>(name, input.ref_, value, output.ref_)
        ->result_tensors()[0];
}

Tensor Model::sub(Tensor input, float value, Tensor output,
                  const std::string &name) {
    return impl_
        ->create_op<ModelOpScalarAdd>(name, input.ref_, -value, output.ref_)
        ->result_tensors()[0];
}

Tensor Model::mul(Tensor input, float value, Tensor output,
                  const std::string &name) {
    return impl_
        ->create_op<ModelOpScalarMul>(name, input.ref_, value, output.ref_)
        ->result_tensors()[0];
}

Tensor Model::div(Tensor input, float value, Tensor output,
                  const std::string &name) {
    return impl_
        ->create_op<ModelOpScalarMul>(name, input.ref_, 1 / value, output.ref_)
        ->result_tensors()[0];
}

}  // namespace ark
