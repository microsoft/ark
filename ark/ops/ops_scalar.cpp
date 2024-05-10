// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "ops_scalar.hpp"

#include "ops_common.hpp"

namespace ark {

ModelOpScalarAssign::ModelOpScalarAssign(float val, const Dims &shape,
                                         ModelDataType data_type,
                                         ModelTensorRef output)
    : ModelOp("ScalarAssign") {
    if (output) {
        check_match_shape(output, shape);
        check_match_data_type(output, data_type);
    } else {
        output = std::make_shared<ModelTensor>(
            data_type, std::make_shared<ModelBuffer>(), shape);
    }
    ModelTensorRef result = std::make_shared<ModelTensor>(*output);
    read_tensors_ = {};
    write_tensors_ = {output};
    result_tensors_ = {result};
    args_ = {{"Value", val}};
    verify();
}

std::string ModelOpScalarAssign::impl_name(const Json &config) const {
    if (!config.contains("NumWarps")) {
        ERR(InvalidUsageError, "NumWarps is required for ",
            type()->type_name());
    } else if (!config.contains("Tile")) {
        ERR(InvalidUsageError, "Tile is required for ", type()->type_name());
    }
    int num_warps = config.at("NumWarps");
    auto &tile_shape = config.at("Tile");
    Dims unit_out_dims{tile_shape[0], tile_shape[1]};

    return function_name_string(
        "scalar_assign", {vec_string(write_tensors_[0]->strides().dims4()),
                          vec_string(write_tensors_[0]->shape().dims4()),
                          vec_string(unit_out_dims.dims4()),
                          std::to_string(num_warps), std::to_string(0)});
}

std::vector<ModelOpArg> ModelOpScalarAssign::impl_args([
    [maybe_unused]] const Json &config) const {
    float val = args_.at("Value").value<float>();
    return {result_tensors_[0], val};
}

Json ModelOpScalarAssign::default_config() const {
    Json config;
    config["NumWarps"] = 1;
    config["SramBytes"] = 0;
    const auto &shape = result_tensors_[0]->shape().dims4();
    size_t tile_x;
    size_t tile_y;
    if (shape[2] > shape[3]) {
        tile_x = 64;
        tile_y = 1;
    } else {
        tile_x = 1;
        tile_y = 64;
    }
    config["Tile"] = {tile_x, tile_y};
    size_t num_tasks = shape[0] * shape[1];
    num_tasks *= (shape[2] + tile_x - 1) / tile_x;
    num_tasks *= (shape[3] + tile_y - 1) / tile_y;
    config["NumTasks"] = num_tasks;
    return config;
}

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
    [maybe_unused]] const Json &config) const {
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
    [maybe_unused]] const Json &config) const {
    float factor = args_.at("Factor").value<float>();
    return {result_tensors_[0], read_tensors_[0], factor};
}

Tensor Model::constant(float val, const Dims &shape, DataType data_type,
                       const std::string &name) {
    return impl_
        ->create_op<ModelOpScalarAssign>(name, val, shape, data_type.ref(),
                                         nullptr)
        ->result_tensors()[0];
}

Tensor Model::copy(float val, Tensor output, const std::string &name) {
    if (output == NullTensor) {
        return impl_
            ->create_op<ModelOpScalarAssign>(name, val, Dims{1}, FP32.ref(),
                                             output.ref())
            ->result_tensors()[0];
    } else {
        return impl_
            ->create_op<ModelOpScalarAssign>(name, val, output.shape(),
                                             output.data_type().ref(),
                                             output.ref())
            ->result_tensors()[0];
    }
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
