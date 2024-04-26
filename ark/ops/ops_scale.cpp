// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "ops_scale.hpp"

#include "ops_common.hpp"

namespace ark {

ModelOpScale::ModelOpScale(ModelTensorRef input, float val,
                           ModelTensorRef output)
    : ModelOp("Scale") {
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
    args_ = {{"Factor", val}};

    verify();
}

std::string ModelOpScale::impl_name(const nlohmann::json &config) const {
    if (!config.contains("NumWarps")) {
        ERR(InvalidUsageError, "NumWarps is required for Scale");
    } else if (!config.contains("Tile")) {
        ERR(InvalidUsageError, "Tile is required for Scale");
    }
    int num_warps = config["NumWarps"];
    auto &tile_shape = config["Tile"];
    Dims unit_out_dims{tile_shape[0], tile_shape[1]};

    std::vector<std::string> template_args;
    template_args.emplace_back(vec_string(read_tensors_[0]->strides().dims4()));
    template_args.emplace_back(vec_string(read_tensors_[0]->shape().dims4()));
    template_args.emplace_back(
        vec_string(write_tensors_[0]->strides().dims4()));
    template_args.emplace_back(vec_string(write_tensors_[0]->shape().dims4()));
    template_args.emplace_back(vec_string(unit_out_dims.dims4()));
    template_args.emplace_back(std::to_string(num_warps));
    template_args.emplace_back(std::to_string(0));
    return function_name_string("scale", template_args);
}

std::vector<ModelOpArg> ModelOpScale::impl_args(
    [[maybe_unused]] const nlohmann::json &config) const {
    float factor = args_.at("Factor").value<float>();
    std::vector<ModelOpArg> args;
    args.emplace_back(result_tensors_[0]);
    args.emplace_back(read_tensors_[0]);
    args.emplace_back(factor);
    return args;
}

nlohmann::ordered_json ModelOpScale::default_config() const {
    nlohmann::ordered_json config;
    config["NumWarps"] = 1;
    config["SramBytes"] = 0;
    const auto &shape = result_tensors_[0]->shape().dims4();
    size_t tile_x;
    size_t tile_y;
    if (shape[2] > shape[3]) {
        tile_x = shape[2] > 64 ? 64 : shape[2];
        tile_y = 1;
    } else {
        tile_x = 1;
        tile_y = shape[3] > 64 ? 64 : shape[2];
    }
    config["Tile"] = {tile_x, tile_y};
    size_t num_tasks = shape[0] * shape[1];
    num_tasks *= (shape[2] + tile_x - 1) / tile_x;
    num_tasks *= (shape[3] + tile_y - 1) / tile_y;
    config["NumTasks"] = num_tasks;
    return config;
}

ModelTensorRef Model::scale(ModelTensorRef input, float val,
                            ModelTensorRef output, const std::string &name) {
    return impl_->create_op<ModelOpScale>(name, input, val, output)
        ->result_tensors()[0];
}

}  // namespace ark
