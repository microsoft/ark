// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "ops_arithmetic.hpp"

#include "ops_common.hpp"

namespace ark {

ModelOpArithmetic::ModelOpArithmetic(const std::string &type_name,
                                     ModelTensorRef input, ModelTensorRef other,
                                     ModelTensorRef output)
    : ModelOp(type_name) {
    check_match_data_type(input, other);
    if (output) {
        check_match_data_type(input, output);
    }
    Dims output_shape = broadcast_shape(input->shape(), other->shape());
    if (output) {
        check_match_shape(output, output_shape);
    } else {
        output = std::make_shared<ModelTensor>(
            input->data_type(), std::make_shared<ModelBuffer>(), output_shape);
    }
    ModelTensorRef result = std::make_shared<ModelTensor>(*output);

    read_tensors_ = {input, other};
    write_tensors_ = {output};
    result_tensors_ = {result};

    verify();
}

std::string ModelOpArithmetic::impl_name(const nlohmann::json &config) const {
    if (!config.contains("NumWarps")) {
        ERR(InvalidUsageError, "NumWarps is required for ",
            type()->type_name());
    } else if (!config.contains("Tile")) {
        ERR(InvalidUsageError, "Tile is required for ", type()->type_name());
    }
    int num_warps = config["NumWarps"];
    auto &tile_shape = config["Tile"];
    Dims unit_out_dims{tile_shape[0], tile_shape[1]};

    std::vector<std::string> template_args;
    template_args.emplace_back(vec_string(read_tensors_[0]->strides().dims4()));
    template_args.emplace_back(vec_string(read_tensors_[0]->shape().dims4()));
    template_args.emplace_back(vec_string(read_tensors_[1]->strides().dims4()));
    template_args.emplace_back(vec_string(read_tensors_[1]->shape().dims4()));
    template_args.emplace_back(
        vec_string(write_tensors_[0]->strides().dims4()));
    template_args.emplace_back(vec_string(write_tensors_[0]->shape().dims4()));
    template_args.emplace_back(vec_string(unit_out_dims.dims4()));
    template_args.emplace_back(std::to_string(num_warps));
    template_args.emplace_back(std::to_string(0));
    return function_name_string(tolower(type()->type_name()), template_args);
}

std::vector<ModelOpArg> ModelOpArithmetic::impl_args(
    [[maybe_unused]] const nlohmann::json &config) const {
    std::vector<ModelOpArg> args;
    args.emplace_back(result_tensors_[0]);
    args.emplace_back(read_tensors_[0]);
    args.emplace_back(read_tensors_[1]);
    return args;
}

nlohmann::ordered_json ModelOpArithmetic::default_config() const {
    nlohmann::ordered_json config;
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

ModelOpAdd::ModelOpAdd(ModelTensorRef input, ModelTensorRef other,
                       ModelTensorRef output)
    : ModelOpArithmetic("Add", input, other, output) {}

ModelTensorRef Model::add(ModelTensorRef input, ModelTensorRef other,
                          ModelTensorRef output, const std::string &name) {
    return impl_->create_op<ModelOpAdd>(name, input, other, output)
        ->result_tensors()[0];
}

ModelOpMul::ModelOpMul(ModelTensorRef input, ModelTensorRef other,
                       ModelTensorRef output)
    : ModelOpArithmetic("Mul", input, other, output) {}

ModelTensorRef Model::mul(ModelTensorRef input, ModelTensorRef other,
                          ModelTensorRef output, const std::string &name) {
    return impl_->create_op<ModelOpMul>(name, input, other, output)
        ->result_tensors()[0];
}

ModelOpSub::ModelOpSub(ModelTensorRef input, ModelTensorRef other,
                       ModelTensorRef output)
    : ModelOpArithmetic("Sub", input, other, output) {}

ModelTensorRef Model::sub(ModelTensorRef input, ModelTensorRef other,
                          ModelTensorRef output, const std::string &name) {
    return impl_->create_op<ModelOpSub>(name, input, other, output)
        ->result_tensors()[0];
}

ModelOpDiv::ModelOpDiv(ModelTensorRef input, ModelTensorRef other,
                       ModelTensorRef output)
    : ModelOpArithmetic("Div", input, other, output) {}

ModelTensorRef Model::div(ModelTensorRef input, ModelTensorRef other,
                          ModelTensorRef output, const std::string &name) {
    return impl_->create_op<ModelOpDiv>(name, input, other, output)
        ->result_tensors()[0];
}

}  // namespace ark
