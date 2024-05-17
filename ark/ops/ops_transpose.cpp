// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "ops_transpose.hpp"

#include <sstream>

#include "logging.h"
#include "ops_common.hpp"

namespace ark {

static std::string permutation_str(const Dims &permutation) {
    int x = DIMS_LEN - permutation.ndims();
    std::vector<DimType> perm4;
    for (int i = 0; i < x; ++i) {
        perm4.push_back(i);
    }
    for (int i = 0; i < permutation.ndims(); ++i) {
        perm4.push_back(permutation[i] + x);
    }
    std::stringstream ss;
    for (auto p : perm4) {
        ss << p;
    }
    return ss.str();
}

static Dims permuted_shape(const Dims &shape, const Dims &permutation) {
    std::vector<DimType> ret;
    for (int i = 0; i < permutation.ndims(); ++i) {
        ret.push_back(shape[permutation[i]]);
    }
    return Dims(ret);
}

ModelOpTranspose::ModelOpTranspose(ModelTensorRef input,
                                   const std::vector<int64_t> &permutation,
                                   ModelTensorRef output)
    : ModelOp("Transpose") {
    check_null(input);
    Dims perm(permutation);
    if (output) {
        check_match_data_type(input, output);
    } else {
        output = std::make_shared<ModelTensor>(
            input->data_type(), std::make_shared<ModelBuffer>(),
            permuted_shape(input->shape(), perm));
    }
    int ndims = input->shape().ndims();
    if (ndims != perm.ndims()) {
        ERR(InvalidUsageError,
            "The number of dimensions of permutation should be the same as "
            "the number of dimensions of input. Given input shape: ",
            input->shape(), ", permutation: ", perm);
    }
    std::vector<int> count(ndims, 0);
    for (int i = 0; i < ndims; ++i) {
        if (perm[i] >= ndims) {
            ERR(InvalidUsageError,
                "Each value in permutation should be less than the number of "
                "input dimensions. Given permutation: ",
                perm);
        }
        if (count[perm[i]] > 0) {
            ERR(InvalidUsageError,
                "Each value in permutation should be unique. Given "
                "permutation: ",
                perm);
        }
        count[perm[i]]++;
    }
    ModelTensorRef result = std::make_shared<ModelTensor>(*output);
    read_tensors_ = {input};
    write_tensors_ = {output};
    result_tensors_ = {result};
    args_.emplace("Permutation", perm);
    verify();
}

std::string ModelOpTranspose::impl_name(const Json &config) const {
    auto permutation = args_.at("Permutation").value<Dims>();
    auto perm_str = permutation_str(permutation);
    int num_warps = config["NumWarps"];
    auto &tile_shape = config["Tile"];
    Dims unit_out_dims{tile_shape[0], tile_shape[1]};
    if (tile_shape[0] < 0) unit_out_dims[0] = write_tensors_[0]->strides()[-2];
    if (tile_shape[1] < 0) unit_out_dims[1] = write_tensors_[0]->strides()[-1];

    return function_name_string(
        "transpose" + perm_str,
        {
            vec_string(read_tensors_[0]->strides().dims4()),
            vec_string(write_tensors_[0]->strides().dims4()),
            vec_string(write_tensors_[0]->shape().dims4()),
            vec_string(unit_out_dims.dims4()),
            std::to_string(num_warps),
            std::to_string(0),
        });
}

std::vector<ModelOpArg> ModelOpTranspose::impl_args([
    [maybe_unused]] const Json &config) const {
    return {result_tensors_[0], read_tensors_[0]};
}

Json ModelOpTranspose::default_config([
    [maybe_unused]] const ArchRef arch) const {
    Json config;
    config["NumWarps"] = 1;
    config["SramBytes"] = 0;
    size_t tile_x = 8;
    size_t tile_y = 8;
    config["Tile"] = {tile_x, tile_y};
    const auto &shape = result_tensors_[0]->shape().dims4();
    size_t num_tasks = shape[0] * shape[1];
    num_tasks *= (shape[2] + tile_x - 1) / tile_x;
    num_tasks *= (shape[3] + tile_y - 1) / tile_y;
    config["NumTasks"] = num_tasks;
    return config;
}

Tensor Model::transpose(Tensor input, const std::vector<int64_t> &permutation,
                        Tensor output, const std::string &name) {
    return impl_
        ->create_op<ModelOpTranspose>(name, input.ref_, permutation,
                                      output.ref_)
        ->result_tensors()[0];
}

}  // namespace ark
