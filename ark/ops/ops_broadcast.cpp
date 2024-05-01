// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "ops_broadcast.hpp"

#include "ops_common.hpp"

namespace ark {

ModelOpBroadcast1::ModelOpBroadcast1(const std::string &type_name,
                                     ModelTensorRef input,
                                     ModelTensorRef output)
    : ModelOp(type_name) {
    check_none(input);
    check_none(output);
    check_match_shape(output, broadcast_shape(input->shape(), output->shape()));
    ModelTensorRef result = std::make_shared<ModelTensor>(*output);

    read_tensors_ = {input};
    write_tensors_ = {output};
    result_tensors_ = {result};

    verify();
}

std::string ModelOpBroadcast1::impl_name(const nlohmann::json &config) const {
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
        tolower(type()->type_name()),
        {vec_string(read_tensors_[0]->strides().dims4()),
         vec_string(read_tensors_[0]->shape().dims4()),
         vec_string(write_tensors_[0]->strides().dims4()),
         vec_string(write_tensors_[0]->shape().dims4()),
         vec_string(unit_out_dims.dims4()), std::to_string(num_warps),
         std::to_string(0)});
}

std::vector<ModelOpArg> ModelOpBroadcast1::impl_args(
    [[maybe_unused]] const nlohmann::json &config) const {
    return {result_tensors_[0], read_tensors_[0]};
}

nlohmann::ordered_json ModelOpBroadcast1::default_config() const {
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

}  // namespace ark
