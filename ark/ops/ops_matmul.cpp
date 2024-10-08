// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "ops_matmul.hpp"

#include <utility>

#include "ops_common.hpp"
#include "utils/utils_math.hpp"

namespace ark {

static Dims calc_problem_size(const Dims &input_shape, const Dims &other_shape,
                              bool trans_input, bool trans_other) {
    int input_ndims = input_shape.ndims();
    int other_ndims = other_shape.ndims();

    if (input_ndims < 1 || other_ndims < 1) ERR(InternalError, "unexpected");

    DimType m;
    DimType n;
    DimType k;
    DimType k2;

    m = (input_ndims == 1) ? 1 : input_shape[-2];
    n = (other_ndims == 1) ? 1 : other_shape[-1];
    k = input_shape[-1];
    k2 = (other_ndims == 1) ? other_shape[0] : other_shape[-2];

    if (trans_input) {
        // Input is column-major
        std::swap(m, k);
    }
    if (trans_other) {
        // Other is column-major
        std::swap(n, k2);
    }
    if (k != k2) {
        ERR(ModelError, "padded inner dimensions mismatch: ", k, " and ", k2);
    }
    return {m, n, k};
}

static Dims calc_output_shape(const Dims &input_shape, const Dims &other_shape,
                              bool trans_input, bool trans_other) {
    // For m, n
    Dims mnk =
        calc_problem_size(input_shape, other_shape, trans_input, trans_other);
    int max_ndims = std::max(input_shape.ndims(), other_shape.ndims());
    if (max_ndims < 3) {
        return {mnk[0], mnk[1]};
    }
    // Considering 4-dimensional matrix multiplication between [N,C,H,W] format
    // tensors, `*_dim_nc` represents the [N,C] value according to the tensor
    // shape. If the tensor is 3-dimensional ([C,H,W]), N is set to 1.
    // If the tensor is 2-dimensional ([H,W]), both N and C are set to 1.
    Dims input_shape_dims4 = input_shape.dims4();
    Dims other_shape_dims4 = other_shape.dims4();
    Dims input_dim_nc{input_shape_dims4[0], input_shape_dims4[1]};
    Dims other_dim_nc{other_shape_dims4[0], other_shape_dims4[1]};
    // Broadcasted output
    Dims output_dim_nc = broadcast_shape(input_dim_nc, other_dim_nc);
    Dims output_shape;
    if (max_ndims == 4) {
        output_shape = {output_dim_nc[0], output_dim_nc[1], mnk[0], mnk[1]};
    } else {  // max_ndims == 3
        output_shape = {output_dim_nc[1], mnk[0], mnk[1]};
    }
    return output_shape;
}

ModelOpMatmul::ModelOpMatmul(ModelTensorRef input, ModelTensorRef other,
                             ModelTensorRef output, bool trans_input,
                             bool trans_other)
    : ModelOp("Matmul") {
    Dims output_shape = calc_output_shape(input->shape(), other->shape(),
                                          trans_input, trans_other);
    Dims padded_output_shape = calc_output_shape(
        input->padded_shape(), other->padded_shape(), trans_input, trans_other);
    // Create an output Tensor.
    if (output) {
        check_match_shape(output, output_shape);
        check_match_padded_shape(output, padded_output_shape);
    } else {
        output = std::make_shared<ModelTensor>(
            input->data_type(), std::make_shared<ModelBuffer>(), output_shape,
            Dims{}, Dims{}, padded_output_shape);
    }
    ModelTensorRef result = std::make_shared<ModelTensor>(*output);

    read_tensors_ = {input, other};
    write_tensors_ = {output};
    result_tensors_ = {result};
    args_["TransposeInput"] = trans_input;
    args_["TransposeOther"] = trans_other;

    verify();
}

std::string ModelOpMatmul::impl_name(const Json &config) const {
    check_fields_config(config, {"NumWarps", "SramBytes", "Tile"});
    check_fields_args(args_, {"TransposeInput", "TransposeOther"});

    bool trans_input = args_.at("TransposeInput").value<bool>();
    bool trans_other = args_.at("TransposeOther").value<bool>();

    const auto &input = read_tensors_[0];
    const auto &other = read_tensors_[1];
    const auto &output = result_tensors_[0];

    check_match_data_type(input, other);
    check_match_data_type(input, output);

    Dims padded_problem_size = calc_problem_size(
        input->padded_shape(), other->padded_shape(), trans_input, trans_other);

    Dims output_shape = calc_output_shape(input->shape(), other->shape(),
                                          trans_input, trans_other);
    Dims padded_output_shape = calc_output_shape(
        input->padded_shape(), other->padded_shape(), trans_input, trans_other);
    check_match_shape(output, output_shape);
    check_match_padded_shape(output, padded_output_shape);

    Dims input_shape_dims4 = input->shape().dims4();
    Dims other_shape_dims4 = other->shape().dims4();
    Dims input_dim_nc{input_shape_dims4[0], input_shape_dims4[1]};
    Dims other_dim_nc{other_shape_dims4[0], other_shape_dims4[1]};
    Dims output_dim_nc = broadcast_shape(input_dim_nc, other_dim_nc);

    Dims strides_acdb{
        input->strides().dims4()[-1], output->strides().dims4()[-1],
        output->strides().dims4()[-1], other->strides().dims4()[-1]};

    int num_warps = config["NumWarps"];
    int smem_bytes = config["SramBytes"];
    Dims tile_shape = config["Tile"].get<std::vector<DimType>>();
    if (tile_shape.ndims() != 2) {
        ERR(PlanError, "Tile should have 2 elements");
    }
    for (int i = 0; i < 2; ++i) {
        if (padded_output_shape[i - 2] % tile_shape[i] != 0) {
            ERR(PlanError, "output padded shape ", padded_output_shape,
                " should be divisible by tile shape ", tile_shape);
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

    DimType size_a = inner_stride_a * output->strides()[-2];
    DimType size_b = inner_stride_b * output->strides()[-1];
    DimType size_c = output->strides()[-2] * output->strides()[-1];
    DimType batch_stride_c_a = input_dim_nc[1] == 1 ? 0 : size_a;
    DimType batch_stride_n_a =
        input_dim_nc[0] == 1 ? 0 : size_a * input_dim_nc[1];
    DimType batch_stride_c_b = other_dim_nc[1] == 1 ? 0 : size_b;
    DimType batch_stride_n_b =
        other_dim_nc[0] == 1 ? 0 : size_b * other_dim_nc[1];
    DimType batch_stride_c_c = output_dim_nc[1] == 1 ? 0 : size_c;
    DimType batch_stride_n_c =
        output_dim_nc[0] == 1 ? 0 : size_c * output_dim_nc[1];
    if (config.contains("BatchStrideNA")) {
        batch_stride_n_a = config["BatchStrideNA"].get<DimType>();
    }
    if (config.contains("BatchStrideNB")) {
        batch_stride_n_b = config["BatchStrideNB"].get<DimType>();
    }
    if (config.contains("BatchStrideNC")) {
        batch_stride_n_c = config["BatchStrideNC"].get<DimType>();
    }
    if (config.contains("BatchStrideCA")) {
        batch_stride_c_a = config["BatchStrideCA"].get<DimType>();
    }
    if (config.contains("BatchStrideCB")) {
        batch_stride_c_b = config["BatchStrideCB"].get<DimType>();
    }
    if (config.contains("BatchStrideCC")) {
        batch_stride_c_c = config["BatchStrideCC"].get<DimType>();
    }

    return function_name_string("matmul",
                                {
                                    vec_string(output->strides().dims4()),
                                    vec_string(input_dim_nc),
                                    vec_string(other_dim_nc),
                                    vec_string(tile_shape),
                                    vec_string(padded_problem_size),
                                    vec_string(strides_acdb),
                                    std::to_string(batch_stride_n_a),
                                    std::to_string(batch_stride_c_a),
                                    std::to_string(batch_stride_n_b),
                                    std::to_string(batch_stride_c_b),
                                    std::to_string(batch_stride_n_c),
                                    std::to_string(batch_stride_c_c),
                                    std::to_string(trans_input),
                                    std::to_string(trans_other),
                                    std::to_string(num_warps),
                                    std::to_string(smem_bytes),
                                });
}

std::vector<ModelOpArg> ModelOpMatmul::impl_args(
    [[maybe_unused]] const Json &config) const {
    return {result_tensors_[0], read_tensors_[0], read_tensors_[1]};
}

static const Json get_default_config(const ArchRef arch,
                                     const ModelDataType &data_type,
                                     const Dims &mnk) {
    if (data_type != FP32.ref() && data_type != FP16.ref() &&
        data_type != BF16.ref()) {
        ERR(PlanError, "Unsupported data type: ", data_type->type_name());
    }
    if (!arch->belongs_to(ARCH_CUDA) && !arch->belongs_to(ARCH_ROCM)) {
        ERR(PlanError, "Unsupported architecture: ", arch->name());
    }
    DimType tm = (mnk[0] > mnk[1]) ? 256 : 128;
    DimType tn = (mnk[0] > mnk[1]) ? 128 : 256;
    if (arch->belongs_to(ARCH_CUDA_80) && data_type == FP32.ref()) {
        return {{"NumWarps", 8}, {"SramBytes", 147456}, {"Tile", {tm, tn}}};
    } else if (arch->belongs_to(ARCH_CUDA_80) && data_type == FP16.ref()) {
        return {{"NumWarps", 8}, {"SramBytes", 147456}, {"Tile", {tm, tn}}};
    } else if (arch->belongs_to(ARCH_CUDA_80) && data_type == BF16.ref()) {
        return {{"NumWarps", 8}, {"SramBytes", 147456}, {"Tile", {tm, tn}}};
    } else if (arch->belongs_to(ARCH_ROCM_942) && data_type == FP32.ref()) {
        return {{"NumWarps", 4}, {"SramBytes", 24672}, {"Tile", {tm, tn}}};
    } else if (arch->belongs_to(ARCH_ROCM_942) && data_type == FP16.ref()) {
        return {{"NumWarps", 4}, {"SramBytes", 24672}, {"Tile", {tm, tn}}};
    } else if (arch->belongs_to(ARCH_ROCM_942) && data_type == BF16.ref()) {
        return {{"NumWarps", 4}, {"SramBytes", 24624}, {"Tile", {tm, tn}}};
    }
    ERR(InternalError, "Unexpected error");
    return {};
}

Json ModelOpMatmul::default_config(const ArchRef arch) const {
    auto result = result_tensors_[0];
    check_fields_args(args_, {"TransposeInput", "TransposeOther"});
    Dims mnk = calc_problem_size(read_tensors_[0]->padded_shape(),
                                 read_tensors_[1]->padded_shape(),
                                 args_.at("TransposeInput").value<bool>(),
                                 args_.at("TransposeOther").value<bool>());
    Json config = get_default_config(arch, result->data_type(), mnk);
    size_t tile_x = config.at("Tile")[0];
    size_t tile_y = config.at("Tile")[1];
    if (mnk[0] % tile_x != 0 || mnk[1] % tile_y != 0) {
        ERR(PlanError, "output padded shape ", Dims{mnk[0], mnk[1]},
            " should be divisible by tile shape ", config.at("Tile"));
    }
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
