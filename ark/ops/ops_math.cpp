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

ModelTensorRef Model::exp(ModelTensorRef input, ModelTensorRef output,
                          const std::string &name) {
    return impl_->create_op<ModelOpExp>(name, input, output)
        ->result_tensors()[0];
}

ModelOpGelu::ModelOpGelu(ModelTensorRef input, ModelTensorRef output)
    : ModelOpMath("Gelu", input, output) {}

ModelTensorRef Model::gelu(ModelTensorRef input, ModelTensorRef output,
                           const std::string &name) {
    return impl_->create_op<ModelOpGelu>(name, input, output)
        ->result_tensors()[0];
}

ModelOpRelu::ModelOpRelu(ModelTensorRef input, ModelTensorRef output)
    : ModelOpMath("Relu", input, output) {}

ModelTensorRef Model::relu(ModelTensorRef input, ModelTensorRef output,
                           const std::string &name) {
    return impl_->create_op<ModelOpRelu>(name, input, output)
        ->result_tensors()[0];
}

ModelOpRsqrt::ModelOpRsqrt(ModelTensorRef input, ModelTensorRef output)
    : ModelOpMath("Rsqrt", input, output) {}

ModelTensorRef Model::rsqrt(ModelTensorRef input, ModelTensorRef output,
                            const std::string &name) {
    return impl_->create_op<ModelOpRsqrt>(name, input, output)
        ->result_tensors()[0];
}

ModelOpSigmoid::ModelOpSigmoid(ModelTensorRef input, ModelTensorRef output)
    : ModelOpMath("Sigmoid", input, output) {}

ModelTensorRef Model::sigmoid(ModelTensorRef input, ModelTensorRef output,
                              const std::string &name) {
    return impl_->create_op<ModelOpSigmoid>(name, input, output)
        ->result_tensors()[0];
}

ModelOpSqrt::ModelOpSqrt(ModelTensorRef input, ModelTensorRef output)
    : ModelOpMath("Sqrt", input, output) {}

ModelTensorRef Model::sqrt(ModelTensorRef input, ModelTensorRef output,
                           const std::string &name) {
    return impl_->create_op<ModelOpSqrt>(name, input, output)
        ->result_tensors()[0];
}

}  // namespace ark
