// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "ops_rope.hpp"

#include "ops_common.hpp"

namespace ark {

ModelOpRope::ModelOpRope(ModelTensorRef input, ModelTensorRef other,
                         ModelTensorRef output)
    : ModelOpBroadcast2("Rope", input, other, output) {}

Json ModelOpRope::default_config([[maybe_unused]] const ArchRef arch) const {
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
    // The rope kernel uses NelemPerThread=2 (complex-multiply on
    // element pairs).  The tile's consecutive dimension must be >= 2
    // so that each pair falls within a single task and vectorized
    // accesses are properly aligned.
    if (tile_y < 2 && shape[3] >= 2) {
        tile_y = 2;
        tile_x = tile_x / 2;
    }
    config["Tile"] = {tile_x, tile_y};
    size_t num_tasks = shape[0] * shape[1];
    num_tasks *= (shape[2] + tile_x - 1) / tile_x;
    num_tasks *= (shape[3] + tile_y - 1) / tile_y;
    config["NumTasks"] = num_tasks;
    return config;
}

Tensor Model::rope(Tensor input, Tensor other, Tensor output,
                   const std::string &name) {
    return impl_
        ->create_op<ModelOpRope>(name, input.ref_, other.ref_, output.ref_)
        ->result_tensors()[0];
}

}  // namespace ark
