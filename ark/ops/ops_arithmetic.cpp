// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "ops_arithmetic.hpp"

#include "ops_common.hpp"

namespace ark {

ModelOpAdd::ModelOpAdd(ModelTensorRef input, ModelTensorRef other,
                       ModelTensorRef output)
    : ModelOpBroadcast2("Add", input, other, output) {}

Tensor Model::add(Tensor input, Tensor other, Tensor output,
                  const std::string &name) {
    return impl_
        ->create_op<ModelOpAdd>(name, input.ref_, other.ref_, output.ref_)
        ->result_tensors()[0];
}

ModelOpMul::ModelOpMul(ModelTensorRef input, ModelTensorRef other,
                       ModelTensorRef output)
    : ModelOpBroadcast2("Mul", input, other, output) {}

Tensor Model::mul(Tensor input, Tensor other, Tensor output,
                  const std::string &name) {
    return impl_
        ->create_op<ModelOpMul>(name, input.ref_, other.ref_, output.ref_)
        ->result_tensors()[0];
}

ModelOpSub::ModelOpSub(ModelTensorRef input, ModelTensorRef other,
                       ModelTensorRef output)
    : ModelOpBroadcast2("Sub", input, other, output) {}

Tensor Model::sub(Tensor input, Tensor other, Tensor output,
                  const std::string &name) {
    return impl_
        ->create_op<ModelOpSub>(name, input.ref_, other.ref_, output.ref_)
        ->result_tensors()[0];
}

ModelOpDiv::ModelOpDiv(ModelTensorRef input, ModelTensorRef other,
                       ModelTensorRef output)
    : ModelOpBroadcast2("Div", input, other, output) {}

Tensor Model::div(Tensor input, Tensor other, Tensor output,
                  const std::string &name) {
    return impl_
        ->create_op<ModelOpDiv>(name, input.ref_, other.ref_, output.ref_)
        ->result_tensors()[0];
}

}  // namespace ark
