// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "ops_matmul.hpp"

#include "ops_common.hpp"
#include "utils/utils_math.hpp"

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

    DimType pm;
    DimType pn;
    DimType pk;
    DimType pk2;

    m = (ndims_a == 1) ? 1 : input->shape()[-2];
    pm = (ndims_a == 1) ? 1 : input->padded_shape()[-2];
    k = input->shape()[-1];
    pk = input->padded_shape()[-1];
    if (trans_input) {
        DimType tmp = m;
        m = k;
        k = tmp;
        tmp = pm;
        pm = pk;
        pk = tmp;
    }
    n = (ndims_b == 1) ? 1 : other->shape()[-1];
    pn = (ndims_b == 1) ? 1 : other->padded_shape()[-1];
    k2 = (ndims_b == 1) ? other->shape()[0] : other->shape()[-2];
    pk2 = (ndims_b == 1) ? other->padded_shape()[0] : other->padded_shape()[-2];
    if (trans_other) {
        DimType tmp = n;
        n = k2;
        k2 = tmp;
        tmp = pn;
        pn = pk2;
        pk2 = tmp;
    }
    if (k != k2) {
        ERR(InvalidUsageError, "inner dimensions mismatch: ", k, " and ", k2);
    }
    if (pk != pk2) {
        ERR(InvalidUsageError, "padded inner dimensions mismatch: ", pk,
            " and ", pk2);
    }

    check_match_data_type(input, other);
    if (output) {
        check_match_data_type(input, output);
    }

    // N and C dimensions of matrix A
    Dims nca{1, 1};
    Dims pnca{1, 1};
    if (ndims_a == 4) {
        nca[0] = input->shape()[0];
        nca[1] = input->shape()[1];
        pnca[0] = input->padded_shape()[0];
        pnca[1] = input->padded_shape()[1];
    } else if (ndims_a == 3) {
        nca[1] = input->shape()[0];
        pnca[1] = input->padded_shape()[0];
    }

    // N and C dimensions of matrix B
    Dims ncb{1, 1};
    Dims pncb{1, 1};
    if (ndims_b == 4) {
        ncb[0] = other->shape()[0];
        ncb[1] = other->shape()[1];
        pncb[0] = other->padded_shape()[0];
        pncb[1] = other->padded_shape()[1];
    } else if (ndims_b == 3) {
        ncb[1] = other->shape()[0];
        pncb[1] = other->padded_shape()[0];
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
    if (pnca[0] != pncb[0] && pnca[0] != 1 && pncb[0] != 1) {
        ERR(InvalidUsageError, "Padded N dimension mismatch: ", pnca[0],
            " and ", pncb[0]);
    }
    if (pnca[1] != pncb[1] && pnca[1] != 1 && pncb[1] != 1) {
        ERR(InvalidUsageError, "Padded C dimension mismatch: ", pnca[1],
            " and ", pncb[1]);
    }

    // N and C dimension of output matrix
    Dims ncc{std::max(nca[0], ncb[0]), std::max(nca[1], ncb[1])};
    Dims pncc{std::max(pnca[0], pncb[0]), std::max(pnca[1], pncb[1])};

    Dims output_shape;
    Dims output_padded_shape;
    if (std::max(ndims_a, ndims_b) == 4) {
        output_shape = {ncc[0], ncc[1], m, n};
        output_padded_shape = {pncc[0], pncc[1], pm, pn};
    } else if (std::max(ndims_a, ndims_b) == 3) {
        output_shape = {ncc[1], m, n};
        output_padded_shape = {pncc[1], pm, pn};
    } else {
        output_shape = {m, n};
        output_padded_shape = {pm, pn};
    }

    // Create an output Tensor.
    if (output) {
        check_match_shape(output, output_shape);
        check_match_padded_shape(output, output_padded_shape);
    } else {
        output = std::make_shared<ModelTensor>(
            input->data_type(), std::make_shared<ModelBuffer>(), output_shape,
            Dims{}, Dims{}, output_padded_shape);
    }
    ModelTensorRef result = std::make_shared<ModelTensor>(*output);

    const Dims &strides_a = input->strides();
    const Dims &strides_b = other->strides();
    const Dims &strides_c = output->strides();

    Dims strides_acdb{strides_a[-1], strides_c[-1], strides_c[-1],
                      strides_b[-1]};

    // a.k.a. problem size
    Dims shape_mnk{m, n, k};
    Dims padded_shape_mnk{pm, pn, pk};

    read_tensors_ = {input, other};
    write_tensors_ = {output};
    result_tensors_ = {result};
    args_["InputDimNC"] = nca;
    args_["OtherDimNC"] = ncb;
    args_["ShapeMNK"] = shape_mnk;
    args_["PaddedShapeMNK"] = padded_shape_mnk;
    args_["StridesACDB"] = strides_acdb;
    args_["TransposeInput"] = trans_input;
    args_["TransposeOther"] = trans_other;

    verify();
}

std::string ModelOpMatmul::impl_name(const Json &config) const {
    if (!config.contains("NumWarps")) {
        ERR(InvalidUsageError, "NumWarps is required");
    } else if (!config.contains("TileShapeMNK")) {
        ERR(InvalidUsageError, "TileShapeMNK is required");
    }

    const auto &input = read_tensors_[0];
    const auto &other = read_tensors_[1];
    const auto &output = result_tensors_[0];

    int num_warps = config["NumWarps"];
    int smem_bytes = config["SramBytes"];
    Dims tile_shape_mnk = config["TileShapeMNK"].get<std::vector<DimType>>();
    Dims input_dim_nc = args_.at("InputDimNC").value<Dims>();
    Dims other_dim_nc = args_.at("OtherDimNC").value<Dims>();
    Dims padded_shape_mnk = args_.at("PaddedShapeMNK").value<Dims>();
    Dims shape_mnk = args_.at("ShapeMNK").value<Dims>();
    Dims strides_acdb = args_.at("StridesACDB").value<Dims>();
    bool trans_input = args_.at("TransposeInput").value<bool>();
    bool trans_other = args_.at("TransposeOther").value<bool>();

    if (tile_shape_mnk.ndims() != 3) {
        ERR(InvalidUsageError, "TileShapeMNK should have 3 elements");
    }

    for (int i = 0; i < 3; ++i) {
        if (padded_shape_mnk[i] % tile_shape_mnk[i] != 0) {
            ERR(InvalidUsageError, "output padded shape MNK ", padded_shape_mnk,
                " should be divisible by tile shape MNK ", tile_shape_mnk);
        }
    }

    DimType inner_stride_a;
    DimType inner_stride_b;
    if (trans_input) {
        inner_stride_a = input->strides().dims4()[-2];
    } else {
        inner_stride_a = input->strides().dims4()[-1];
    }
    if (trans_other) {
        inner_stride_b = other->strides().dims4()[-1];
    } else {
        inner_stride_b = other->strides().dims4()[-2];
    }

    return function_name_string("matmul",
                                {
                                    vec_string(output->strides().dims4()),
                                    vec_string(input_dim_nc),
                                    vec_string(other_dim_nc),
                                    vec_string(tile_shape_mnk),
                                    vec_string(padded_shape_mnk),
                                    vec_string(strides_acdb),
                                    std::to_string(inner_stride_a),
                                    std::to_string(inner_stride_b),
                                    std::to_string(trans_input),
                                    std::to_string(trans_other),
                                    std::to_string(num_warps),
                                    std::to_string(smem_bytes),
                                });
}

std::vector<ModelOpArg> ModelOpMatmul::impl_args([
    [maybe_unused]] const Json &config) const {
    return {result_tensors_[0], read_tensors_[0], read_tensors_[1]};
}

static const Json get_default_config(const ArchRef arch,
                                     const ModelDataType &data_type) {
    if (arch->belongs_to(ARCH_CUDA_80) && data_type == FP32.ref()) {
        return {{"NumWarps", 8},
                {"SramBytes", 147456},
                {"TileShapeMNK", {128, 256, 32}},
                {"TilePadMNK", {128, 256, 32}}};
    } else if (arch->belongs_to(ARCH_CUDA_80) && data_type == FP16.ref()) {
        return {{"NumWarps", 8},
                {"SramBytes", 147456},
                {"TileShapeMNK", {128, 256, 64}},
                {"TilePadMNK", {128, 256, 64}}};
    } else if (arch->belongs_to(ARCH_CUDA_80) && data_type == BF16.ref()) {
        return {{"NumWarps", 8},
                {"SramBytes", 147456},
                {"TileShapeMNK", {128, 256, 64}},
                {"TilePadMNK", {128, 256, 64}}};
    } else if (arch->belongs_to(ARCH_ROCM_942) && data_type == FP32.ref()) {
        return {{"NumWarps", 4},
                {"SramBytes", 24672},
                {"TileShapeMNK", {128, 256, 16}},
                {"TilePadMNK", {128, 256, 16}}};
    } else if (arch->belongs_to(ARCH_ROCM_942) && data_type == FP16.ref()) {
        return {{"NumWarps", 4},
                {"SramBytes", 24672},
                {"TileShapeMNK", {128, 256, 32}},
                {"TilePadMNK", {128, 256, 32}}};
    } else if (arch->belongs_to(ARCH_ROCM_942) && data_type == BF16.ref()) {
        return {{"NumWarps", 4},
                {"SramBytes", 24672},
                {"TileShapeMNK", {128, 256, 32}},
                {"TilePadMNK", {128, 256, 32}}};
    }
    ERR(InvalidUsageError, "Unsupported arch and data type: ", arch->name(),
        " and ", data_type->type_name());
    return {};
}

Json ModelOpMatmul::default_config(const ArchRef arch) const {
    Dims shape_mnk = args_.at("ShapeMNK").value<Dims>();
    Dims input_dim_nc = args_.at("InputDimNC").value<Dims>();
    Dims other_dim_nc = args_.at("OtherDimNC").value<Dims>();
    auto result = result_tensors_[0];
    Json config = get_default_config(arch, result->data_type());
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
