// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "ops_matmul.hpp"

#include "math_utils.h"
#include "ops_common.hpp"

namespace ark {

ModelOpMatmul::ModelOpMatmul(ModelTensorRef input, ModelTensorRef other,
                             ModelTensorRef output, bool trans_input,
                             bool trans_other)
    : ModelOp("Matmul") {
    // Shape verification.
    const Dims &shp_a = input->shape();
    const Dims &shp_b = other->shape();
    int ndims_a = shp_a.ndims();
    int ndims_b = shp_b.ndims();

    if (ndims_a < 1) {
        ERR(InvalidUsageError, "input has an empty shape: ", shp_a);
    }
    if (ndims_b < 1) {
        ERR(InvalidUsageError, "other has an empty shape: ", shp_b);
    }

    // m: the number of rows of output matrix (row-major)
    // n: the number of columns of output matrix (row-major)
    // k: the inner dimension of matrix multiplication
    DimType m;
    DimType n;
    DimType k;
    DimType k2;

    m = (ndims_a == 1) ? 1 : shp_a[-2];
    k = shp_a[-1];
    if (trans_input) {
        DimType tmp = m;
        m = k;
        k = tmp;
    }
    n = (ndims_b == 1) ? 1 : shp_b[-1];
    k2 = (ndims_b == 1) ? shp_b[0] : shp_b[-2];
    if (trans_other) {
        DimType tmp = n;
        n = k2;
        k2 = tmp;
    }
    if (k != k2) {
        ERR(InvalidUsageError, "inner dimensions mismatch: ", k, " and ", k2);
    }

    check_match_data_type(input, other);
    if (output) {
        check_match_data_type(input, output);
    }

    // N and C dimensions of matrix A
    Dims nca{1, 1};
    if (ndims_a == 4) {
        nca[0] = shp_a[0];
        nca[1] = shp_a[1];
    } else if (ndims_a == 3) {
        nca[1] = shp_a[0];
    }

    // N and C dimensions of matrix B
    Dims ncb{1, 1};
    if (ndims_b == 4) {
        ncb[0] = shp_b[0];
        ncb[1] = shp_b[1];
    } else if (ndims_b == 3) {
        ncb[1] = shp_b[0];
    }

    // Verify broadcasting
    if (nca[0] != ncb[0] && nca[0] != 1 && ncb[0] != 1) {
        ERR(InvalidUsageError, "N dimension mismatch: ", nca[0], " and ",
            ncb[0]);
    }
    if (nca[1] != ncb[1] && nca[1] != 1 && ncb[1] != 1) {
        ERR(InvalidUsageError, "C dimension mismatch: ", nca[1], " and ",
            ncb[1]);
    }

    // N and C dimension of output matrix
    Dims ncc{std::max(nca[0], ncb[0]), std::max(nca[1], ncb[1])};

    Dims output_shape;
    if (std::max(ndims_a, ndims_b) == 4) {
        output_shape = {ncc[0], ncc[1], m, n};
    } else if (std::max(ndims_a, ndims_b) == 3) {
        output_shape = {ncc[1], m, n};
    } else {
        output_shape = {m, n};
    }

    // Create an output Tensor.
    if (output) {
        check_match_shape(output, output_shape);
    } else {
        output = std::make_shared<ModelTensor>(
            input->data_type(), std::make_shared<ModelBuffer>(), output_shape);
    }
    ModelTensorRef result = std::make_shared<ModelTensor>(*output);

    const Dims &strides_a = input->strides();
    const Dims &strides_b = other->strides();
    const Dims &strides_c = output->strides();

    Dims strides_acdb{strides_a[-1], strides_c[-1], strides_c[-1],
                      strides_b[-1]};

    // a.k.a. problem size
    Dims shape_mnk{m, n, k};

    read_tensors_ = {input, other};
    write_tensors_ = {output};
    result_tensors_ = {result};
    args_["InputDimNC"] = nca;
    args_["OtherDimNC"] = ncb;
    args_["ShapeMNK"] = shape_mnk;
    args_["StridesACDB"] = strides_acdb;
    args_["IsInputColumnMajor"] = trans_input;
    args_["IsOtherColumnMajor"] = trans_other;

    verify();
}

std::string ModelOpMatmul::impl_name(const nlohmann::json &config) const {
    if (!config.contains("NumWarps")) {
        ERR(InvalidUsageError, "NumWarps is required");
    } else if (!config.contains("TileShapeMNK")) {
        ERR(InvalidUsageError, "TileShapeMNK is required");
    }
    int num_warps = config["NumWarps"];
    int smem_bytes = config["SramBytes"];
    Dims tile_shape_mnk = config["TileShapeMNK"].get<std::vector<DimType>>();
    Dims tile_pad_mnk = config["TilePadMNK"].get<std::vector<DimType>>();
    Dims input_dim_nc = args_.at("InputDimNC").value<Dims>();
    Dims other_dim_nc = args_.at("OtherDimNC").value<Dims>();
    Dims shape_mnk = args_.at("ShapeMNK").value<Dims>();
    Dims strides_acdb = args_.at("StridesACDB").value<Dims>();
    bool is_input_column_major = args_.at("IsInputColumnMajor").value<bool>();
    bool is_other_column_major = args_.at("IsOtherColumnMajor").value<bool>();

    if (tile_shape_mnk.ndims() != tile_pad_mnk.ndims()) {
        ERR(InvalidUsageError,
            "TileShapeMNK and TilePadMNK should have the same "
            "number of dimensions");
    }
    for (auto i = 0; i < tile_shape_mnk.ndims(); i++) {
        if (tile_shape_mnk[i] % tile_pad_mnk[i] != 0) {
            ERR(InvalidUsageError,
                "TileShapeMNK should be divisible by TilePadMNK");
        }
    }

    Dims output_strides = result_tensors_[0]->strides().dims4();
    output_strides[-2] = math::pad(output_strides[-2], tile_pad_mnk[0]);
    output_strides[-1] = math::pad(output_strides[-1], tile_pad_mnk[1]);
    if (output_strides[-2] % tile_pad_mnk[0] != 0) {
        ERR(InvalidUsageError, "output stride ", output_strides[-2],
            " should be divisible by tile pad ", tile_pad_mnk[0]);
    }
    if (output_strides[-1] % tile_pad_mnk[1] != 0) {
        ERR(InvalidUsageError, "output stride ", output_strides[-1],
            " should be divisible by tile pad ", tile_pad_mnk[1]);
    }

    shape_mnk[0] = math::pad(shape_mnk[0], tile_pad_mnk[0]);
    shape_mnk[1] = math::pad(shape_mnk[1], tile_pad_mnk[1]);
    shape_mnk[2] = math::pad(shape_mnk[2], tile_pad_mnk[2]);

    DimType inner_stride_a = shape_mnk[2];
    DimType inner_stride_b = shape_mnk[2];

    return function_name_string("matmul",
                                {
                                    vec_string(output_strides),
                                    vec_string(input_dim_nc),
                                    vec_string(other_dim_nc),
                                    vec_string(tile_shape_mnk),
                                    vec_string(shape_mnk),
                                    vec_string(strides_acdb),
                                    std::to_string(inner_stride_a),
                                    std::to_string(inner_stride_b),
                                    std::to_string(is_input_column_major),
                                    std::to_string(is_other_column_major),
                                    std::to_string(num_warps),
                                    std::to_string(smem_bytes),
                                });
}

std::vector<ModelOpArg> ModelOpMatmul::impl_args(
    [[maybe_unused]] const nlohmann::json &config) const {
    return {result_tensors_[0], read_tensors_[0], read_tensors_[1]};
}

nlohmann::ordered_json ModelOpMatmul::default_config() const {
    Dims shape_mnk = args_.at("ShapeMNK").value<Dims>();
    Dims input_dim_nc = args_.at("InputDimNC").value<Dims>();
    Dims other_dim_nc = args_.at("OtherDimNC").value<Dims>();
    auto result = result_tensors_[0];
    nlohmann::ordered_json config;
    if (result->data_type() == FP32.ref()) {
        config["NumWarps"] = 4;
        config["SramBytes"] = 49152;
        config["TileShapeMNK"] = {64, 64, 32};
        config["TilePadMNK"] = {64, 64, 32};
    } else if (result->data_type() == FP16.ref()) {
        config["NumWarps"] = 4;
        config["SramBytes"] = 98304;
        config["TileShapeMNK"] = {64, 64, 64};
        config["TilePadMNK"] = {64, 64, 64};
    } else if (result->data_type() == BF16.ref()) {
        config["NumWarps"] = 4;
        config["SramBytes"] = 98304;
        config["TileShapeMNK"] = {64, 64, 64};
        config["TilePadMNK"] = {64, 64, 64};
    }
    size_t tile_x = config.at("TileShapeMNK")[0];
    size_t tile_y = config.at("TileShapeMNK")[1];
    size_t num_tasks = std::max(input_dim_nc[0], other_dim_nc[0]);
    num_tasks *= std::max(input_dim_nc[1], other_dim_nc[1]);
    num_tasks *= (shape_mnk[0] + tile_x - 1) / tile_x;
    num_tasks *= (shape_mnk[1] + tile_y - 1) / tile_y;
    config["NumTasks"] = num_tasks;
    return config;
}

Tensor Model::matmul(Tensor input, Tensor other, Tensor output,
                     bool trans_input, bool trans_other,
                     const std::string &name) {
    return impl_
        ->create_op<ModelOpMatmul>(name, input.ref(), other.ref(), output.ref(),
                                   trans_input, trans_other)
        ->result_tensors()[0];
}

}  // namespace ark
