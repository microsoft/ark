// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "ops_math.hpp"

#include "ops_common.hpp"

namespace ark {

ModelOpMath::ModelOpMath(const std::string &type_name, ModelTensorRef input,
                         ModelTensorRef output)
    : ModelOpBroadcast1(
          type_name, input,
          output ? output
                 : std::make_shared<ModelTensor>(
                       input->data_type(), std::make_shared<ModelBuffer>(),
                       input->shape())) {
    if (output) {
        check_match_data_type(input, output);
    }
    verify();
}

ModelOpExp::ModelOpExp(ModelTensorRef input, ModelTensorRef output)
    : ModelOpMath("Exp", input, output) {}

Tensor Model::exp(Tensor input, Tensor output, const std::string &config,
                  const std::string &name) {
    return impl_->create_op<ModelOpExp>(config, name, input.ref_, output.ref_)
        ->result_tensors()[0];
}

ModelOpGelu::ModelOpGelu(ModelTensorRef input, ModelTensorRef output)
    : ModelOpMath("Gelu", input, output) {}

Tensor Model::gelu(Tensor input, Tensor output, const std::string &config,
                   const std::string &name) {
    return impl_->create_op<ModelOpGelu>(config, name, input.ref_, output.ref_)
        ->result_tensors()[0];
}

ModelOpRelu::ModelOpRelu(ModelTensorRef input, ModelTensorRef output)
    : ModelOpMath("Relu", input, output) {}

Tensor Model::relu(Tensor input, Tensor output, const std::string &config,
                   const std::string &name) {
    return impl_->create_op<ModelOpRelu>(config, name, input.ref_, output.ref_)
        ->result_tensors()[0];
}

ModelOpRsqrt::ModelOpRsqrt(ModelTensorRef input, ModelTensorRef output)
    : ModelOpMath("Rsqrt", input, output) {}

Tensor Model::rsqrt(Tensor input, Tensor output, const std::string &config,
                    const std::string &name) {
    return impl_->create_op<ModelOpRsqrt>(config, name, input.ref_, output.ref_)
        ->result_tensors()[0];
}

ModelOpSigmoid::ModelOpSigmoid(ModelTensorRef input, ModelTensorRef output)
    : ModelOpMath("Sigmoid", input, output) {}

Tensor Model::sigmoid(Tensor input, Tensor output, const std::string &config,
                      const std::string &name) {
    return impl_
        ->create_op<ModelOpSigmoid>(config, name, input.ref_, output.ref_)
        ->result_tensors()[0];
}

ModelOpSqrt::ModelOpSqrt(ModelTensorRef input, ModelTensorRef output)
    : ModelOpMath("Sqrt", input, output) {}

Tensor Model::sqrt(Tensor input, Tensor output, const std::string &config,
                   const std::string &name) {
    return impl_->create_op<ModelOpSqrt>(config, name, input.ref_, output.ref_)
        ->result_tensors()[0];
}

}  // namespace ark
