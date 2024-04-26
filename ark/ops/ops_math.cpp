// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "ops_math.hpp"

#include "ops_common.hpp"

namespace ark {

ModelOpMath::ModelOpMath(const std::string &type_name, ModelTensorRef input,
                         ModelTensorRef output)
    : ModelOp(type_name) {
    if (output) {
        check_match_data_type(input, output);
        check_match_shape(input, output);
    } else {
        output = std::make_shared<ModelTensor>(input->data_type(),
                                               std::make_shared<ModelBuffer>(),
                                               input->shape());
    }
    ModelTensorRef result = std::make_shared<ModelTensor>(*output);

    read_tensors_ = {input};
    write_tensors_ = {output};
    result_tensors_ = {result};

    verify();
}

ModelOpExp::ModelOpExp(ModelTensorRef input, ModelTensorRef output)
    : ModelOpMath("Exp", input, output) {}

ModelTensorRef Model::exp(ModelTensorRef input, ModelTensorRef output,
                          const std::string &name) {
    return impl_->create_op<ModelOpExp>(name, input, output)
        ->result_tensors()[0];
}

ModelOpRelu::ModelOpRelu(ModelTensorRef input, ModelTensorRef output)
    : ModelOpMath("Relu", input, output) {}

ModelTensorRef Model::relu(ModelTensorRef input, ModelTensorRef output,
                           const std::string &name) {
    return impl_->create_op<ModelOpRelu>(name, input, output)
        ->result_tensors()[0];
}

}  // namespace ark
